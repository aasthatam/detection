import os
import torch
from torch.utils.data import DataLoader
from data_class import FashionTripletDataset
from triplets import TripletNet
from ml.triplets import triplet_loss

def train_model(num_epochs=20, batch_size=64, learning_rate=1e-4, margin=1.0, subset_size=40000):
    # Load datasets with subset
    train_dataset = FashionTripletDataset(
        txt_file="dataset/fashiondataset_triplet_train_with_hard.txt",
        base_dir="dataset",
        subset_size=subset_size  # Use e.g. 10,000 training examples
    )

    val_dataset = FashionTripletDataset(
        txt_file="dataset/fashiondataset_triplet_test_with_hard.txt", 
        base_dir="dataset",
        subset_size=2000  # Use e.g. 2,000 validation examples
    )

    # Increase batch size and workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # For model saving
    best_val_loss = float('inf')
    
    # Training Loop with test set as validation
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for i, (anchor, pos, neg) in enumerate(train_loader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)
            loss = triplet_loss(emb_a, emb_p, emb_n, margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Validation (using test set)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for anchor, pos, neg in val_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                
                emb_a = model(anchor)
                emb_p = model(pos)
                emb_n = model(neg)
                loss = triplet_loss(emb_a, emb_p, emb_n, margin)
                
                total_val_loss += loss.item()
        
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, f'checkpoints/best_model_epoch_{epoch+1}.pt')
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pt')
    
    return model
