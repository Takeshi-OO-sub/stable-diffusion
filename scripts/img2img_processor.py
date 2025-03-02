import os
import torch
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
import numpy as np
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
from safetensors.torch import load_file

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    sd = load_file(ckpt)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def process_image(input_path, output_path, model, sampler):
    # 画像の読み込みと前処理
    init_image = Image.open(input_path).convert("RGB")
    init_image = init_image.resize((512, 512))
    init_image = np.array(init_image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).cuda()
    init_image = 2.0 * init_image - 1.0

    # latent spaceへの変換
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

    # 生成パラメータの設定
    sampler.make_schedule(ddim_num_steps=150, ddim_eta=0.0, verbose=False)
    
    # ノイズの追加
    t_enc = int(150 * 0.5)
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).cuda())
    
    # 条件付き生成
    c = model.get_learned_conditioning([""])  # 空の文字列で無条件生成を指定
    samples = sampler.decode(z_enc, 
                           c,  # 条件付き生成用のテンソル
                           t_start=t_enc,
                           unconditional_guidance_scale=15.0,
                           unconditional_conditioning=c)  # 同じ形状のテンソルを使用

    # latent spaceから画像への変換
    x_samples = model.decode_first_stage(samples)

    # 画像の後処理と保存
    x_samples = ((x_samples + 1.0) / 2.0).clamp(0.0, 1.0)
    x_samples = torch.nn.functional.interpolate(
        x_samples, size=(init_image.shape[2], init_image.shape[3]), mode="bilinear"
    )
    x_samples = x_samples.cpu().numpy().transpose(0, 2, 3, 1)[0]
    x_samples = (255 * x_samples).astype(np.uint8)
    Image.fromarray(x_samples).save(output_path)

def main():
    # モデルの設定とロード
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(
        config,
        "ldm/models/diffusion/realisticVisionV60B1_v51HyperVAE.safetensors"
    )
    sampler = DDIMSampler(model)

    # 入力・出力フォルダの設定
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 入力フォルダ内の画像を処理
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(input_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_image(input_path, output_path, model, sampler)

if __name__ == "__main__":
    main() 