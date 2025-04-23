import { Photo, PhotoGroup, PhotoMetadata } from '../types';
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  Tensor,
  cos_sim,
  matmul,
} from '@huggingface/transformers';
import ExifReader from 'ExifReader';

const positiveQualityPrompts = [
  "a high-quality photo",
  "a sharp photo",
  "a clear photo",
  "a well-lit photo",
  "a vibrant photo",
  "a professional photo",
  "a happy photo",
  "a cool photo",
  "a classy photo",
  "a smiling person"
];

const negativeQualityPrompts = [
  "a low-quality photo",
  "a blurry photo",
  "an out-of-focus photo",
  "a dark photo",
  "a noisy photo",
  "an amateur photo",
  "a sad photo",
  "a poorly composed photo",
  "a bad photo",
  "a messy background"
];

/* ---------- Model lazy‑loading ----------------------------------------- */
let visionModelPromise: Promise<InstanceType<typeof CLIPVisionModelWithProjection>> | null = null;
let processorPromise: Promise<InstanceType<typeof AutoProcessor>> | null = null;
let textModelPromise: Promise<InstanceType<typeof CLIPTextModelWithProjection>> | null = null;
let tokenizerPromise: Promise<InstanceType<typeof AutoTokenizer>> | null = null;

async function getModels() {
  if (!visionModelPromise) {
    const modelId = 'jinaai/jina-clip-v1';
    const processorId = 'xenova/clip-vit-base-patch32';

    processorPromise = processorPromise || AutoProcessor.from_pretrained(processorId, { device: 'webgpu' });
    visionModelPromise = visionModelPromise || CLIPVisionModelWithProjection.from_pretrained(modelId);
    tokenizerPromise = tokenizerPromise || AutoTokenizer.from_pretrained(modelId);
    textModelPromise = textModelPromise || CLIPTextModelWithProjection.from_pretrained(modelId);
  }
  const [processor, vision_model, tokenizer, text_model] = await Promise.all([
    processorPromise!,
    visionModelPromise!,
    tokenizerPromise!,
    textModelPromise!,
  ]);
  return { processor, vision_model, tokenizer, text_model };
}

export async function extractFeatures(photo: Photo): Promise<number[]> {
  const { processor, vision_model } = await getModels();
  const image = await RawImage.read(photo.file);
  const image_inputs = await (processor as any)([image]);
  const { image_embeds } = await vision_model(image_inputs);
  return Array.from(image_embeds.normalize(2, -1).data);
}

export async function prepareQualityEmbeddings(): Promise<{
  positiveEmbeddings: Tensor;
  negativeEmbeddings: Tensor;
}> {
  const { tokenizer, text_model } = await getModels();
  const allPrompts = [...positiveQualityPrompts, ...negativeQualityPrompts];

  const text_inputs = (tokenizer as any)(allPrompts, { padding: true, truncation: true });
  const { text_embeds } = await text_model(text_inputs);

  const normedEmbeds = text_embeds.normalize(2, -1);

  const nPos = positiveQualityPrompts.length;
  const nNeg = negativeQualityPrompts.length;

  const positiveEmbeddings = normedEmbeds.slice(
    [0, nPos],
    [0, normedEmbeds.dims[1]]
  );
  const negativeEmbeddings = normedEmbeds.slice(
    [nPos, nPos + nNeg],
    [0, normedEmbeds.dims[1]]
  );
  text_model.dispose();
  return { positiveEmbeddings, negativeEmbeddings };
}

/* ---------- Quality + metadata analysis -------------------------------- */
export async function analyzeImage(
  file: File,
  url: string,
  embedding: number[] | null,
  qualityEmbeddings: { positiveEmbeddings: Tensor; negativeEmbeddings: Tensor } | null
): Promise<{ quality: number; metadata: PhotoMetadata }> {

  let quality = 0;

  if (embedding && embedding.length > 0 && qualityEmbeddings) {
    try {
      const v = new Tensor(
        'float32',
        Float32Array.from(embedding),
        [1, embedding.length]
      );

      const { positiveEmbeddings: P, negativeEmbeddings: N } = qualityEmbeddings;

      const simPos = await matmul(v, P.transpose(1, 0)); // [1, k_pos]
      const simNeg = await matmul(v, N.transpose(1, 0)); // [1, k_neg]

      const avgPos = simPos.sum().div(P.dims[0]).item() as number;
      const avgNeg = simNeg.sum().div(N.dims[0]).item() as number;

      // simple linear calibration (placeholder)
      const rawScore = avgPos - avgNeg;
      quality = Math.max(0, Math.min(100, Math.round(((rawScore * 15 + 1) / 2) * 100)));

    } catch (error) {
      console.error("Error calculating image quality:", error);
      quality = 0;
    }
  } else {
    console.warn("Embedding or quality embeddings not provided, setting quality to 0.");
    quality = 0;
  }

  const arrayBuffer = await file.arrayBuffer();
  const exifTags = ExifReader.load(arrayBuffer);
  console.log('exif', exifTags)

  const dateFromExif = exifTags?.['DateTimeOriginal']?.description || exifTags?.['DateTime']?.description

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const metadata: PhotoMetadata = {
        width: img.width,
        height: img.height,
        captureDate: new Date(dateFromExif?.replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3") || file.lastModified),
      };
      resolve({ quality, metadata });
    };
    img.onerror = () => {
      console.error("Error loading image for metadata:", url);
      const metadata: PhotoMetadata = { captureDate: new Date(file.lastModified) };
      resolve({ quality, metadata });
    };
    img.src = url;
  });
}

/* ---------- Deduplication ---------------------------------------------- */
export function groupSimilarPhotos(
  photos: Photo[],
  similarityThreshold: number = 0.7
): { groups: PhotoGroup[]; uniquePhotos: Photo[] } {
  const photosWithEmbeddings = photos.filter(p => p.embedding && p.embedding.length > 0);
  if (photosWithEmbeddings.length === 0) {
    const photosWithoutEmbeddings = photos.filter(p => !p.embedding || p.embedding.length === 0);
    photosWithoutEmbeddings.sort((a, b) => b.dateCreated.getTime() - a.dateCreated.getTime());
    return { groups: [], uniquePhotos: photosWithoutEmbeddings };
  }

  const groups: PhotoGroup[] = [];
  const uniquePhotos: Photo[] = [];
  const processed = new Set<string>();

  for (let i = 0; i < photosWithEmbeddings.length; i++) {
    const photoA = photosWithEmbeddings[i];
    if (processed.has(photoA.id)) continue;

    const currentGroupPhotos: Photo[] = [photoA];
    let minSimilarityInGroup = 1.0;
    processed.add(photoA.id);

    for (let j = i + 1; j < photosWithEmbeddings.length; j++) {
      const photoB = photosWithEmbeddings[j];
      if (processed.has(photoB.id)) continue;

      if (photoA.embedding && photoB.embedding) {
        const similarity = cos_sim(photoA.embedding, photoB.embedding);
        if (similarity >= similarityThreshold) {
          currentGroupPhotos.push(photoB);
          minSimilarityInGroup = Math.min(minSimilarityInGroup, similarity);
          processed.add(photoB.id);
        }
      }
    }

    if (currentGroupPhotos.length > 1) {
      const sortedPhotos = [...currentGroupPhotos].sort((a, b) => (b.quality ?? 0) - (a.quality ?? 0));
      groups.push({
        id: `${sortedPhotos[0].id}-group`,
        title: getGroupTitle(sortedPhotos[0]),
        date: sortedPhotos[0].dateCreated,
        photos: sortedPhotos,
        similarity: minSimilarityInGroup,
        similarityThreshold,
      });
    } else {
      uniquePhotos.push(photoA);
    }
  }

  // ensure photos lacking embeddings are preserved
  photos.forEach(p => {
    if ((!p.embedding || p.embedding.length === 0) && !uniquePhotos.some(up => up.id === p.id)) {
      uniquePhotos.push(p);
    }
  });

  groups.sort((a, b) => b.date.getTime() - a.date.getTime());
  uniquePhotos.sort((a, b) => b.dateCreated.getTime() - a.dateCreated.getTime());
  return { groups, uniquePhotos };
}

/* ---------- Utility helpers -------------------------------------------- */
function getGroupTitle(photo: Photo): string {
  const date = photo.metadata!.captureDate!;
  const formattedDate = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const formattedTime = date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
  return `${formattedDate} at ${formattedTime}`;
}
