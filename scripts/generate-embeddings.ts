import { writeFile } from 'fs/promises';
import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  Tensor,
} from '@huggingface/transformers';

const positivePrompts = [
  'a high-quality photo', 'a sharp photo', 'a clear photo', 'a well-lit photo',
  'a vibrant photo', 'a professional photo', 'a happy photo',
  'a cool photo', 'a classy photo', 'a smiling person'
];
const negativePrompts = [
  'a low-quality photo', 'a blurry photo', 'an out-of-focus photo',
  'a dark photo', 'a noisy photo', 'an amateur photo', 'a sad photo',
  'a poorly composed photo', 'a bad photo', 'a messy background'
];

const MODEL_ID = 'jinaai/jina-clip-v1';

function tensorToNested(t: Tensor): number[][] {
  const [rows, cols] = t.dims;
  const out: number[][] = [];
  const data = t.data as Float32Array;
  for (let r = 0; r < rows; r++) {
    out.push(Array.from(data.slice(r * cols, (r + 1) * cols)));
  }
  return out;
}

(async () => {
  console.log('⏬  downloading text tower…');
  const tokenizer   = await AutoTokenizer.from_pretrained(MODEL_ID);
  const textModel   = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { device: 'cpu' });

  const input       = tokenizer([...positivePrompts, ...negativePrompts],
                                { padding: true, truncation: true });
  const { text_embeds } = await textModel(input);
  const normed      = text_embeds.normalize(2, -1);

  const nPos = positivePrompts.length;

  const pos = normed.slice([0, nPos],              [0, normed.dims[1]]);
  const neg = normed.slice([nPos, nPos + negativePrompts.length], [0, normed.dims[1]]);

  const json = {
    positive: tensorToNested(pos),
    negative: tensorToNested(neg)
  };

  await writeFile('src/data/qualityEmbeds.json', JSON.stringify(json));
  console.log('✅  Wrote src/data/qualityEmbeds.json');
})();
