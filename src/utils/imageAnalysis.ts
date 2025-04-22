import { Photo, PhotoGroup, PhotoMetadata } from '../types';
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  Tensor,
  cos_sim,
  env
} from '@huggingface/transformers';

env.allowLocalModels=false;
env.allowRemoteModels=true;

// Load models and processor once (promise-based lazy init)
let visionModelPromise: Promise<InstanceType<typeof CLIPVisionModelWithProjection>> | null = null;
let processorPromise: Promise<InstanceType<typeof AutoProcessor>> | null = null;
let textModelPromise: Promise<InstanceType<typeof CLIPTextModelWithProjection>> | null = null;
let tokenizerPromise: Promise<InstanceType<typeof AutoTokenizer>> | null = null;

// Define specific quality prompts
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

async function getModels() {
  if (!visionModelPromise) {
    const modelId = 'jinaai/jina-clip-v1';
    const processorId = 'xenova/clip-vit-base-patch32';

    processorPromise = AutoProcessor.from_pretrained(processorId, { device: 'webgpu'});
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore: from_pretrained expects only one argument for this API
    visionModelPromise = CLIPVisionModelWithProjection.from_pretrained(modelId);
    tokenizerPromise = AutoTokenizer.from_pretrained(modelId);
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore: from_pretrained expects only one argument for this API
    textModelPromise = CLIPTextModelWithProjection.from_pretrained(modelId);
  }
  const [processor, vision_model, tokenizer, text_model] = await Promise.all([
    processorPromise!,
    visionModelPromise!,
    tokenizerPromise!,
    textModelPromise!,
  ]);
  return { processor, vision_model, tokenizer, text_model };
}

/**
 * Extracts image features (embeddings) using a CLIP model.
 * Uses the File object if available (recommended for uploaded images).
 */
export async function extractFeatures(photo: Photo): Promise<number[]> {
  const { processor, vision_model } = await getModels();
  const image = await RawImage.read(photo.file);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const image_inputs = await (processor as any)([image]);
  const { image_embeds } = await vision_model(image_inputs);
  const embeddingData = image_embeds.data as Float32Array;
  return Array.from(embeddingData);
}

/**
 * Prepares text embeddings for quality analysis prompts.
 */
export async function prepareQualityEmbeddings(): Promise<{ positiveEmbeddings: Tensor, negativeEmbeddings: Tensor }> {
  const { tokenizer, text_model } = await getModels();
  const allPrompts = [...positiveQualityPrompts, ...negativeQualityPrompts];

  // Assuming the tokenizer instance is callable directly
  // @ts-expect-error: AutoTokenizer type doesn't declare call signature but is callable
  const text_inputs = tokenizer(allPrompts, { padding: true, truncation: true });

  const { text_embeds } = await text_model(text_inputs);

  // Split the embeddings based on the number of positive prompts

const nPos = positiveQualityPrompts.length;  // 10
const nNeg = negativeQualityPrompts.length;  // 10

// rows 0 (inclusive) → nPos (exclusive), all columns 0→embedDim
const positiveEmbeddings = text_embeds.slice(
  [ 0,    nPos ],    // slice rows [0..10)
  [ 0,    text_embeds.dims[1] ] // slice cols [0..768)
);

// rows nPos→nPos+nNeg ([10..20)), all columns
const negativeEmbeddings = text_embeds.slice(
  [ nPos, nPos + nNeg ], // slice rows [10..20)
  [ 0,    text_embeds.dims[1] ]     // slice cols [0..768)
);
  return { positiveEmbeddings, negativeEmbeddings };
}

/**
 * Analyzes an image file to extract basic metadata and calculate quality based on embedding.
 * Requires the image embedding and pre-calculated quality prompt embeddings.
 */
export async function analyzeImage(
  file: File,
  url: string,
  embedding: number[] | null,
  qualityEmbeddings: { positiveEmbeddings: Tensor, negativeEmbeddings: Tensor } | null
): Promise<{ quality: number; metadata: PhotoMetadata }> {

  let quality = 0;

  if (embedding && embedding.length > 0 && qualityEmbeddings) {
    try {
      const { positiveEmbeddings, negativeEmbeddings } = qualityEmbeddings;

      // Calculate average similarity to positive prompts
      let totalPositiveSimilarity = 0;
      for (let i = 0; i < positiveEmbeddings.dims[0]; ++i) {
          const positiveEmbeddingSlice = positiveEmbeddings.slice([i, i + 1], [0, positiveEmbeddings.dims[1]]);
          const positiveEmbeddingData = Array.from(positiveEmbeddingSlice.data as Float32Array);
          totalPositiveSimilarity += cos_sim(embedding, positiveEmbeddingData);
      }
      const avgPositiveSimilarity = totalPositiveSimilarity / positiveEmbeddings.dims[0];


      // Calculate average similarity to negative prompts
      let totalNegativeSimilarity = 0;
      for (let i = 0; i < negativeEmbeddings.dims[0]; ++i) {
          const negativeEmbeddingSlice = negativeEmbeddings.slice([i, i + 1], [0, negativeEmbeddings.dims[1]]);
          const negativeEmbeddingData = Array.from(negativeEmbeddingSlice.data as Float32Array);
          totalNegativeSimilarity += cos_sim(embedding, negativeEmbeddingData);
      }
      const avgNegativeSimilarity = totalNegativeSimilarity / negativeEmbeddings.dims[0];


      // Combine similarities into a 0-100 score
      // Maps the difference (range approx -1 to 1) to 0-100
      const rawScore = avgPositiveSimilarity - avgNegativeSimilarity;
      quality = Math.max(0, Math.min(100, Math.round(((rawScore*15 + 1) / 2) * 100)));


    } catch (error) {
      console.error("Error calculating image quality:", error);
      quality = 0; // Default to 0 on error
    }
  } else {
    console.warn("Embedding or quality embeddings not provided, setting quality to 0.");
    quality = 0;
  }

  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const metadata: PhotoMetadata = {
        width: img.width,
        height: img.height,
        captureDate: new Date(file.lastModified),
        camera: getCameraFromFileName(file.name),
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

/**
 * Groups similar photos based on CLIP image embeddings.
 * Assumes `photo.embedding` has been populated beforehand by calling `extractFeatures`.
 */
export function groupSimilarPhotos(photos: Photo[], similarityThreshold: number = 0.7): { groups: PhotoGroup[]; uniquePhotos: Photo[] } {
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
      const sortedPhotos = [...currentGroupPhotos].sort((a, b) => b.quality! - a.quality!);
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
  photos.forEach(p => {
    if ((!p.embedding || p.embedding.length === 0) && !uniquePhotos.some(up => up.id === p.id)) {
      uniquePhotos.push(p);
    }
  });
  groups.sort((a, b) => b.date.getTime() - a.date.getTime());
  uniquePhotos.sort((a, b) => b.dateCreated.getTime() - a.dateCreated.getTime());
  return { groups, uniquePhotos };
}

// Helper for group title
function getGroupTitle(photo: Photo): string {
  const date = photo.dateCreated;
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

// Helper for camera name
function getCameraFromFileName(filename: string): string {
  const lowerFilename = filename.toLowerCase();
  if (lowerFilename.startsWith('img_')) return 'iPhone';
  if (lowerFilename.startsWith('dsc_') || lowerFilename.startsWith('dscf')) return 'Nikon/Fuji';
  if (lowerFilename.startsWith('img-')) return 'Android/Other';
  if (lowerFilename.startsWith('p_') || lowerFilename.startsWith('p')) return 'Huawei/Pixel';
  if (lowerFilename.endsWith('.heic')) return 'Apple Device';
  if (lowerFilename.includes('canon')) return 'Canon';
  if (lowerFilename.includes('sony')) return 'Sony';
  if (/\d{8}_\d{6}/.test(filename)) return 'Android/Other';
  return 'Unknown Camera';
}