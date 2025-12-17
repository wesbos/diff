import { pipeline, cos_sim } from "@huggingface/transformers";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Initialize the image feature extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/bge-base-en-v1.5"
);


// Get image paths from command line or use defaults
const args = process.argv.slice(2);
const image1Path = args[0] || path.join(__dirname, "images", "jake1.png");
const image2Path = args[1] || path.join(__dirname, "images", "jake2.png");

console.log("🖼️  Comparing images:");
console.log(`   Image 1: ${path.basename(image1Path)}`);
console.log(`   Image 2: ${path.basename(image2Path)}`);
console.log("");

// Extract features from both images
console.log("⏳ Extracting features...");

const [features1, features2] = await Promise.all([
  extractor(image1Path, { pooling: "mean" }),
  extractor(image2Path, { pooling: "mean" }),
]);

// Get the embedding vectors (flatten to 1D arrays)
const embedding1 = Array.from(features1.data as Float32Array);
const embedding2 = Array.from(features2.data as Float32Array);

console.log(`   Vector dimensions: ${embedding1.length}`);
console.log("");

// Calculate cosine similarity using transformers.js built-in function
const similarity = cos_sim(embedding1, embedding2);

// Convert to percentage (cosine similarity ranges from -1 to 1)
// For normalized embeddings, it's typically 0 to 1
const percentage = Math.max(0, similarity) * 100;

console.log("📊 Results:");
console.log(`   Cosine Similarity: ${similarity.toFixed(6)}`);
console.log(`   Match Percentage:  ${percentage.toFixed(2)}%`);
console.log("");

// Interpret the result
if (percentage >= 95) {
  console.log("✅ Nearly identical images!");
} else if (percentage >= 80) {
  console.log("🟢 Very similar images");
} else if (percentage >= 60) {
  console.log("🟡 Moderately similar images");
} else if (percentage >= 40) {
  console.log("🟠 Somewhat similar images");
} else {
  console.log("🔴 Different images");
}
