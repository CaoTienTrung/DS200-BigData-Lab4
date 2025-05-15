import time
import json
import socket
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomImageDataset
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Stream images from a custom dataset via socket to Spark Streaming'
)
parser.add_argument('--directory', '-d', required=True, type=str,
                    help='Root directory of the image dataset')
parser.add_argument('--split', '-s', choices=['train', 'dev', 'test'], default='train',
                    help='Select train, dev or test split')
parser.add_argument('--batch_size', '-b', required=True, type=int,
                    help='Batch size')
parser.add_argument('--endless', '-e', required=False, type=bool, default=False,
                    help='Enable endless streaming')
parser.add_argument('--label_mode', '-l', choices=['int','binary','categorical', None], default='int',
                    help='Label mode for dataset')
parser.add_argument('--color_mode', '-c', choices=['rgb','grayscale'], default='rgb',
                    help='Color mode for images')
parser.add_argument('--image_size', '-i', nargs=2, type=int, default=(256,256),
                    help='Resize images to this size (H W)')
parser.add_argument('--interpolation', '-p', choices=['nearest','bilinear'], default='bilinear',
                    help='Interpolation method for resize')
parser.add_argument('--sleep', '-t', required=False, type=int, default=3,
                    help='Sleep time between batches (seconds)')
args = parser.parse_args()

# --- TCP configuration ---
TCP_IP   = "localhost"
TCP_PORT = 6100

class Streamer:
    def __init__(self, dataset, batch_size, split, sleep_time):
        self.batch_size = batch_size
        self.split = split
        self.sleep_time = sleep_time
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )

    def connect_TCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"[Streamer] Waiting for connection on {TCP_IP}: {TCP_PORT} …")
        conn, addr = s.accept()
        print(f"[Streamer] Connected to {addr}")
        self.conn = conn
        return conn, addr

    def send_batch(self, batch_imgs, batch_labels):
        arr = batch_imgs.numpy().reshape(batch_imgs.size(0), -1).tolist()
        payload = {}
        data_sent = 0
        for i, row in enumerate(arr):
            feat = {f'feature-{j}': row[j] for j in range(len(row))}
            lbl = batch_labels[i].item() if hasattr(batch_labels[i], 'item') else batch_labels[i]
            payload[i] = {**feat, 'label':lbl}
            # print(payload[i])
        msg = (json.dumps(payload) + '\n').encode('utf-8')
        try:
            self.conn.sendall(msg)
        except BrokenPipeError:
            print("Connection lost.")
            return False
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
            return False
        return True

    def stream_dataset(self):
        pbar = tqdm(self.loader)
        for idx, (imgs, labels) in tqdm(enumerate(self.loader)):
            ok = self.send_batch(imgs, labels)
            if not ok:
                break
            pbar.update(n=1)

            time.sleep(self.sleep_time)

if __name__ == '__main__':
    dataset = CustomImageDataset(
        directory=os.path.join(args.directory, args.split),
        label_mode=args.label_mode,
        color_mode=args.color_mode,
        image_size=tuple(args.image_size),
        interpolation=args.interpolation
        )
    streamer = Streamer(dataset, args.batch_size, args.split, args.sleep)
    streamer.connect_TCP()

    if args.endless == True:
        while True:
            streamer.stream_dataset()
    else:
        streamer.stream_dataset()

    streamer.conn.close()


