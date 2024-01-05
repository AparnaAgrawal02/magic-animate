# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import datetime
import inspect
import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import torch.distributed as dist

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.util import save_videos_grid
from magicanimate.utils.dist_tools import distributed_init
from accelerate.utils import set_seed

from magicanimate.utils.videoreader import VideoReader

from einops import rearrange

from pathlib import Path


def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    config  = OmegaConf.load(args.config)
      
    # Initialize distributed training
    device = torch.device(f"cuda:{args.rank}")
    dist_kwargs = {"rank":args.rank, "world_size":args.world_size, "dist":args.dist}
    
    if config.savename is None:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path(args.config).stem}-{time_str}"
    else:
        savedir = f"samples/{config.savename}"
        
    if args.dist:
        dist.broadcast_object_list([savedir], 0)
        dist.barrier()
    
    if args.rank == 0:
        os.makedirs(savedir, exist_ok=True)

    inference_config = OmegaConf.load(config.inference_config)
        
    motion_module = config.motion_module
    
    ### >>> create animation pipeline >>> ###
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    if config.pretrained_unet_path:
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
    else:
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
    appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path, subfolder="appearance_encoder").to(device)
    reference_control_writer = ReferenceAttentionControl(appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
    reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
    if config.pretrained_vae_path is not None:
        vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
    else:
        vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

    ### Load controlnet
    controlnet   = ControlNetModel.from_pretrained(config.pretrained_controlnet_path)

    unet.enable_xformers_memory_efficient_attention()
    appearance_encoder.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()

    vae.to(torch.float16)
    unet.to(torch.float16)
    text_encoder.to(torch.float16)
    appearance_encoder.to(torch.float16)
    controlnet.to(torch.float16)

    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        # NOTE: UniPCMultistepScheduler
    )
    #pipeline.enable_vae_slicing()
    

    # 1. unet ckpt
    # 1.1 motion module
    motion_module_state_dict = torch.load(motion_module, map_location="cpu")
    if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
    motion_module_state_dict = motion_module_state_dict['state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
    try:
        # extra steps for self-trained models
        state_dict = OrderedDict()
        for key in motion_module_state_dict.keys():
            if key.startswith("module."):
                _key = key.split("module.")[-1]
                state_dict[_key] = motion_module_state_dict[key]
            else:
                state_dict[key] = motion_module_state_dict[key]
        motion_module_state_dict = state_dict
        del state_dict
        missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0
    except:
        _tmp_ = OrderedDict()
        for key in motion_module_state_dict.keys():
            if "motion_modules" in key:
                if key.startswith("unet."):
                    _key = key.split('unet.')[-1]
                    _tmp_[_key] = motion_module_state_dict[key]
                else:
                    _tmp_[key] = motion_module_state_dict[key]
        missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
        assert len(unexpected) == 0
        del _tmp_
    del motion_module_state_dict

    pipeline.to(device)
    ### <<< create validation pipeline <<< ###
    
    random_seeds = config.get("seed", [-1])
    random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
    
    # input test videos (either source video/ conditions)
    #open ../../video_label.py and ../../ label_curr-count_req-count.txt
    #randomly select 
    #print pwd 
   # print("current working directory: ", os.getcwd())
    videolists = pd.read_csv("video_label.txt", sep="\t")
    print(videolists)
    print(videolists.columns)
    # breakpoint()
    count = videolists["label"].value_counts()
    labels = count.keys().tolist()
    dict1 = {}
    test_videos = []
    source_images = []
    parallel_labels = []
    if not os.path.exists("videos_selected.txt"):
        for x in labels:
            to_make = 10

            if os.path.exists("/ssd_scratch/cvit/aparna/magicanimate/samples/wasl4/videos/"+str(x)):
                to_make -= len(os.listdir("/ssd_scratch/cvit/aparna/magicanimate/samples//wasl4/videos/"+str(x)))
        # print("to_make: ", to_make, "count[x]: ", count[x],x)
            if to_make > 0:
                # print(videolists)
                # print("dfdfd",videolists["label"])
                # print(videolists[videolists["label"] == x])
                videos_to =videolists[videolists["label"] == x].sample(to_make, replace=True)
                videos = videos_to["video"].tolist()
                test_videos.extend(videos)
                parallel_labels.extend([x]*len(videos))
                dict1[x] = videos
        #save dict1 t o a file
        pd.DataFrame.from_dict(dict1, orient='index').to_csv("videos_selected.txt", sep="\t", header=False)
    #choose len(test_videos) number of videos from videolist and take first image as source image and save in source_images
    #print("test_videos: ", test_videos)
    else:
        dict1 = pd.read_csv("videos_selected.txt", sep="\t", header=None)
        dict1 = dict1.to_dict()
        dict1 = dict1[0]
        for x in dict1:
            test_videos.extend(dict1[x].split(","))
    random_seeds = random_seeds * len(test_videos) if len(random_seeds) == 1 else random_seeds
    #5 len name 
    if len(str(x)) < 5:
        x = "0"*(5-len(str(x)))+str(x)
    
    test_videos = [os.path.join("/ssd_scratch/cvit/aparna/dataset/WLASL/video/train",str(x)+".mp4") for x in test_videos]
    all_videos = videolists["video"].tolist()
    print("all_videos: ", all_videos)
   # breakpoint()
    if not os.path.exists("source_images"):
        os.mkdir("source_images")
    for x in test_videos:
        print(x)
        x = x.split("/")[-1].split(".")[0]
        print(x)
        #breakpoint()
        all_videos = videolists["video"].tolist()
        all_videos.remove(int(x))
        selected = random.choice(all_videos)
        video = os.path.join("/ssd_scratch/cvit/aparna/dataset/WLASL/video/train", str(selected)+".mp4")
        print("video: ", video)
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        if not ret:
            print("error in reading video")
            continue

        cv2.imwrite("source_images/"+x+"_"+str(selected)+".jpg", frame)
        source_images.append("source_images/"+x+"_"+str(selected)+".jpg")
        cap.release()
        cv2.destroyAllWindows()


    
    print("test_videos: ", test_videos)
    print("source_images: ", source_images)
    # breakpoint()
    # test_videos = videolists["video"].tolist()
    # source_images = config.source_image
    num_actual_inference_steps = config.get("num_actual_inference_steps", config.steps)

    # read size, step from yaml file
    sizes = [config.size] * len(test_videos)
    steps = [config.S] * len(test_videos)

    config.random_seed = []
    prompt = n_prompt = ""
    for idx, (source_image, test_video, random_seed, size, step,label) in tqdm(
        enumerate(zip(source_images, test_videos, random_seeds, sizes, steps,parallel_labels)), 
        total=len(test_videos), 
        disable=(args.rank!=0)
    ):
        v_name = test_video.split("/")[-1].split(".")[0]
        if len(str(v_name)) < 5:
            v_name = "0"*(5-len(str(v_name)))+str(v_name)
        test_video = os.path.join("/ssd_scratch/cvit/aparna/dataset/WLASL/video/train",str(v_name)+".mp4")
        print(f"Processing {test_video} ...")
        print(f"current seed: {torch.initial_seed()}")
        print(f"image: {source_image}")
        samples_per_video = []
        samples_per_clip = []
        # manually set random seed for reproduction
        if random_seed != -1: 
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()
        config.random_seed.append(torch.initial_seed())

        if test_video.endswith('.mp4'):
            control = VideoReader(test_video).read()
            if control[0].shape[0] != size:
                control = [np.array(Image.fromarray(c).resize((size, size))) for c in control]
            if config.max_length is not None:
                control = control[config.offset: (config.offset+config.max_length)]
            control = np.array(control)
        
        if source_image.endswith(".mp4"):
            source_image = np.array(Image.fromarray(VideoReader(source_image).read()[0]).resize((size, size)))
        else:
            source_image = np.array(Image.open(source_image).resize((size, size)))
        H, W, C = source_image.shape
        
        print(f"current seed: {torch.initial_seed()}")
        init_latents = None
        
        # print(f"sampling {prompt} ...")
        original_length = control.shape[0]
        if control.shape[0] % config.L > 0:
            control = np.pad(control, ((0, config.L-control.shape[0] % config.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())
        sample = pipeline(
            prompt,
            negative_prompt         = n_prompt,
            num_inference_steps     = config.steps,
            guidance_scale          = config.guidance_scale,
            width                   = W,
            height                  = H,
            video_length            = len(control),
            controlnet_condition    = control,
            init_latents            = init_latents,
            generator               = generator,
            num_actual_inference_steps = num_actual_inference_steps,
            appearance_encoder       = appearance_encoder, 
            reference_control_writer = reference_control_writer,
            reference_control_reader = reference_control_reader,
            source_image             = source_image,
            **dist_kwargs,
        ).videos

        if args.rank == 0:
            source_images1 = np.array([source_image] * original_length)
            source_images1 = rearrange(torch.from_numpy(source_images1), "t h w c -> 1 c t h w") / 255.0
            samples_per_video.append(source_images1)
            
            control = control / 255.0
            control = rearrange(control, "t h w c -> 1 c t h w")
            control = torch.from_numpy(control)
            samples_per_video.append(control[:, :, :original_length])

            samples_per_video.append(sample[:, :, :original_length])
                
            samples_per_video = torch.cat(samples_per_video)

            video_name = os.path.basename(test_video)[:-4]
            source_name = os.path.basename(source_images[idx]).split(".")[0]
            print(f"Saving video {video_name} ...")
            save_videos_grid(samples_per_video[-1:], f"{savedir}/videos/{label}/{source_name}_{video_name}.mp4")
            #save_videos_grid(samples_per_video, f"{savedir}/videos/{source_name}_{video_name}/grid.mp4")

            if config.save_individual_videos:
                save_videos_grid(samples_per_video[1:2], f"{savedir}/videos/{source_name}_{video_name}/ctrl.mp4")
                save_videos_grid(samples_per_video[0:1], f"{savedir}/videos/{source_name}_{video_name}/orig.mp4")
                
        if args.dist:
            dist.barrier()
               
    if args.rank == 0:
        OmegaConf.save(config, f"{savedir}/config.yaml")


def distributed_main(device_id, args):
    args.rank = device_id
    args.device_id = device_id
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()
    distributed_init(args)
    main(args)


def run(args):

    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        assert args.world_size <= torch.cuda.device_count()

        if args.world_size > 0 and torch.cuda.device_count() > 1:
            port = random.randint(10000, 20000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
    else:
        main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dist", action="store_true", required=False)
    parser.add_argument("--rank", type=int, default=0, required=False)
    parser.add_argument("--world_size", type=int, default=1, required=False)

    args = parser.parse_args()
    run(args)
