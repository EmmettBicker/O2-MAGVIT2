# -*- coding:UTF-8 -*-
import torch
import os
from modeling.magvit_model import VisionTokenizer
import torchvision.transforms as T
from src.utils import get_config
from PIL import Image
from argparse import ArgumentParser

from torch.utils.data import Dataset, DataLoader


def save_tokens_to_file(tokens, output_path):
    tokens = tokens.cpu()
    torch.save(tokens, output_path)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, target_pixels, spatial_downsample_ratio):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        frames_torch = T.functional.pil_to_tensor(img).unsqueeze(dim=0)
        frames = frames_torch.float() / 127.5 - 1  # Basic normalization
        frames = frames.unsqueeze(dim=2)
        return frames


def concatenate_and_maxpad(tensors, pad_value=-1):
    """
    Concatenates and pads a list of 1D tensors to the maximum length among them.

    Parameters:
        tensors (list of torch.Tensor): List of 1D tensors to concatenate and pad.
        pad_value (int, optional): The value to use for padding. Defaults to -1.

    Returns:
        torch.Tensor: A 2D tensor where all rows are padded to the same length.
    """
    # Find the maximum length of the tensors
    max_length = max(tensor.size(1) for tensor in tensors)
    # Pad tensors to the maximum length with the specified padding value
    padded_tensors = [torch.nn.functional.pad(tensor, (0, max_length - tensor.size(1)), value=pad_value) for tensor in tensors]

    # Stack the padded tensors into a 2D tensor
    return torch.stack(padded_tensors, dim=0)


def process_image_dataset(data_dir, output_dir, model, tokenizer, batch_size, target_pixels, spatial_downsample_ratio, start_idx):
    dataset = ImageFolderDataset(data_dir, target_pixels=None, spatial_downsample_ratio=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    all_tokens = []
    with torch.no_grad():
        for batch_idx, frames in enumerate(dataloader):
            if batch_idx < start_idx:
                continue
            if frames.numel() > (3*1000*1000):
                print(f"Skipped frame of shape {frames.shape} and index: ", batch_idx)
                continue

            try:
                frames = frames.squeeze(1)
                frames = frames.to("cuda")
                _, encoded_output, *_ = tokenizer.encode(frames, entropy_loss_weight=0.0)
                token_ids = encoded_output.indices
                all_tokens.append(token_ids.cpu())

                if (batch_idx + 1) % 100 == 0:
                    print(f"Processed {batch_idx + 1} images")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                raise Exception(f"Error at batch {batch_idx}")

            # Save every N images
            if len(all_tokens) >= 1000:
                batch_tokens = concatenate_and_maxpad(all_tokens)
                save_tokens_to_file(
                    batch_tokens, 
                    f"{output_dir}/image_tokens_{start_idx}_{batch_idx}.pt"
                )

                all_tokens = []

    # Save remaining tokens
    if all_tokens:
        batch_tokens = concatenate_and_maxpad(all_tokens)
        save_tokens_to_file(
            batch_tokens, 
            f"{output_dir}/image_tokens_{start_idx}_{batch_idx}.pt"
        )


def main(args=None):
    if args is None:
        args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model and config
    states = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model_config = get_config(args.config_path)
    model = VisionTokenizer(
        config=model_config, 
        commitment_cost=0, 
        diversity_gamma=0, 
        use_gan=False, 
        use_lecam_ema=False, 
        use_perceptual=False
    )
    model.tokenizer.load_state_dict(states, strict=True)
    tokenizer = model.tokenizer
    model.eval()
    model.to("cuda")

    spatial_downsample_ratio = 2 ** (len(model_config.model.decoder.channel_multipliers) - 1)

    # Process image dataset with batching
    process_image_dataset(
        data_dir=args.image_dir,
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        batch_size=1,  # Adjust based on your GPU memory
        target_pixels=args.target_pixels,
        spatial_downsample_ratio=spatial_downsample_ratio,
        start_idx=args.start_idx
    )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--target-pixels", type=int, default=384)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    return parser.parse_args()

  
if __name__ == "__main__":
    main()