import torch
import cv2
import os
import sys
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from segment_anything import SamPredictor
from PIL import Image
import glob
import argparse
from tqdm import tqdm

HOME = os.getcwd()


GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(f"{HOME}/GroundingDINO", "weights", "groundingdino_swint_ogc.pth")
GROUNDING_DINO_CONFIG_PATH = os.path.join(f"{HOME}/GroundingDINO", "groundingdino/config/GroundingDINO_SwinT_OGC.py")
SAM_CHECKPOINT_PATH = os.path.join(f"{HOME}/GroundingDINO", "weights", "sam_vit_h_4b8939.pth")

SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOX_TRESHOLD = 0.25     #0.35 for all else | 0.30 for football 0.15 for softball and tennis
TEXT_TRESHOLD = 0.25    #0.25


def load_models():
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return sam_predictor, grounding_dino_model

def add_all_suffix(class_names):
    result = []
    for class_name in class_names:
        result.append("all " + class_name + "s")
    return result

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
        #result_masks.append(masks[0])
    return np.array(result_masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Dataset path", default='data')
    args = parser.parse_args()
    class_names = {
        #'basketball': ['all humans', 'basketball'], 
        'boxes': ['human', 'packages'], 
        #'football': ['all humans', 'oval ball'],
        #'juggle': ['human', 'juggle balls'], 
        #'softball': ['human', 'bat'],
        #'tennis': ['human', 'racket']
    }
    captions = {
        #'basketball': ['two people passing a basketball'], 
        'boxes': 'a person with boxes', 
        #'football': 'people passing a football',
        #'juggle': 'a person juggling balls', 
        #'softball': 'a person holding a bat',
        #'tennis': 'a person swinging a racket'
    }
    
    sam_predictor, grounding_dino_model = load_models()

    for seq in class_names.keys():
        print(class_names[seq])
        for i in range(1, 31):
            # load image
            input_path = f"{args.dataset_path}/{seq}/ims/{i}"
            images = glob.glob(os.path.join(input_path, "*.jpg"))
            images = sorted(images)
            for k, image_path in tqdm(enumerate(images), total=len(images)):
                # detect objects
                image = np.array(Image.open(image_path))
                # detections = grounding_dino_model.predict_with_classes(
                #     image=image,
                #     classes=class_names[seq],
                #     box_threshold=BOX_TRESHOLD,
                #     text_threshold=TEXT_TRESHOLD
                # )

                # detections.mask = segment(
                #     sam_predictor=sam_predictor,
                #     image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                #     xyxy=detections.xyxy
                # )

                # mask = np.sum(detections.mask * 255, axis=0)
                
                detections = grounding_dino_model.predict_with_caption(
                    image=image,
                    caption=captions[seq],
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )

                mask = segment(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections[0].xyxy
                )

                mask = np.sum(mask * 255, axis=0)
                data_dir = "data"

                if not os.path.exists(os.path.join(args.dataset_path, f"{seq}/sam_mask/{i}")):
                    os.makedirs(os.path.join(os.path.join(args.dataset_path, f"{seq}/sam_mask/{i}")))

                save_path = os.path.join(data_dir, f"{seq}/sam_mask/{i}", str(k).zfill(6) + ".jpg")
                Image.fromarray(mask.astype(np.uint8)).save(save_path)