# Fashion Image Similarity Search System

This system provides a fashion product image similarity search capability for the e-commerce platform. It consists of:

1. A pre-trained TripletNet model for generating fashion image embeddings
2. A FastAPI service to handle image processing and similarity searches
3. Qdrant vector database for efficient similarity retrieval
4. Integration with the Node.js e-commerce backend

## System Architecture

- **TripletNet Model**: A trained ResNet50-based neural network that converts fashion images into 128-dimensional embeddings
- **FastAPI Server**: Handles image processing, embedding generation, and similarity searches
- **Qdrant**: Vector database for storing and querying embeddings
- **Node.js Backend Integration**: Automatically stores product image embeddings when new products are added

## Setup Instructions

### 1. Start Qdrant in Docker

First, start the Qdrant vector database using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

This will start Qdrant on ports 6333 (API) and 6334 (web UI). The web UI is accessible at http://localhost:6334.

### 2. FastAPI Service Setup

```bash
# Install dependencies
cd detection/app
pip install -r requirements.txt

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

By default, the FastAPI service will connect to Qdrant at localhost:6333. If your Qdrant is running on a different host or port, you can set environment variables:

```bash
QDRANT_HOST=your-qdrant-host QDRANT_PORT=6333 uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Node.js Backend Integration

Make sure to install axios in the backend:

```bash
cd backend-fashion
npm install axios
```

The `productController.js` has been updated to communicate with the FastAPI service when products are added or removed, and to handle similarity searches.

### 4. Importing Existing Products

To import embeddings for all existing products:

```bash
cd detection/app
python import_existing_products.py
```

## API Endpoints

### FastAPI Endpoints (localhost:8000)

- `/store_embedding` - Store embedding for a single product image
- `/store_multiple_embeddings` - Batch store multiple embeddings
- `/upload_image` - Upload an image and store its embedding
- `/find_similar` - Find similar products 
- `/remove_embedding` - Remove an embedding

### Node.js Endpoints

- `/api/products/similar` - Find similar products (integrates with FastAPI)

## How It Works

1. When an admin uploads product images, the Node.js backend stores them in Cloudinary
2. After saving the product to MongoDB, it sends the image URLs to the FastAPI service
3. The FastAPI service generates embeddings using the TripletNet model and stores them in Qdrant
4. When users search for similar products, the backend queries the FastAPI service
5. The service finds the most similar products based on cosine similarity of the embeddings

## Requirements

### Python
- Python 3.8+
- FastAPI, Uvicorn
- PyTorch, TorchVision
- Qdrant client
- Pillow
- Requests

### Node.js
- Node.js 18+
- Express
- Axios
- Cloudinary
- Mongoose

### Docker
- Docker Engine for running Qdrant

## Credits

The TripletNet model was trained using a triplet loss approach on fashion image data to learn how to generate embeddings that place similar fashion items close together in the embedding space. 