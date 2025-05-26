from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils

from networks.DeflickerNet import DeflickerNet
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DeflickerNet')
    parser.add_argument('--method', type=str, default="DFVG")
    ### dataset options
    parser.add_argument('--dataset', type=str,default="ship", help='dataset to test')
    parser.add_argument('--phase', type=str, default="test", choices=["train", "test"])
    parser.add_argument('--data_dir', type=str, default='data', help='path to data folder')
    parser.add_argument('--list_dir', type=str, default='lists', help='path to list folder')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint folder')
    ### other options
    parser.add_argument('-gpu', type=int, default=0, help='gpu device id')

    args = parser.parse_args()
    args.cuda = True

    args.size_multiplier = 2 ** 2

    print(args)

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")

    ### load model opts
    opts_filename = os.path.join(args.checkpoint_dir, args.method, "opts.pth")
    print("Load %s" % opts_filename)
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)

    ### initialize model
    print('===> Initializing model from %s...' % model_opts.model)
    model = DeflickerNet(model_opts, nc_in=12, nc_out=3)

    ### load trained model
    model_filename = os.path.join(args.checkpoint_dir, args.method, "model.pth")
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename,map_location='cuda:0')
    model.load_state_dict(state_dict['model'])

    ### convert to GPU
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)

    model.eval()

    ### load video list
    list_filename = os.path.join(args.list_dir, "%s_%s.txt" % (args.dataset, args.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    ### start testing
    for v in range(len(video_list)):

        video = video_list[v]

        print("Test on %s-%s video %d/%d: %s" % (args.dataset, args.phase, v + 1, len(video_list), video))

        ## setup path
        input_dir = os.path.join(args.data_dir, args.phase, "input", video)
        process_dir = os.path.join(args.data_dir, args.phase, "output/filter", video)
        output_dir = os.path.join(args.data_dir, args.phase, "output/final_depth", video)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = glob.glob(os.path.join(input_dir, "*.jpg"))
        output_list = glob.glob(os.path.join(output_dir, "*.jpg"))

        if len(frame_list) == len(output_list) and not args.redo:
            print("Output frames exist, skip...")
            continue

        ## frame 0
        frame_p1 = utils.read_img(os.path.join(process_dir, "000001.png"))
        output_filename = os.path.join(output_dir, "000001.png")
        utils.save_img(frame_p1, output_filename)

        lstm_state = None

        for t in tqdm(range(1, len(frame_list))):
            ### load frames
            frame_i1 = utils.read_img(os.path.join(input_dir, "%06d.jpg" % (t)))
            frame_i2 = utils.read_img(os.path.join(input_dir, "%06d.jpg" % (t+1)))
            frame_o1 = utils.read_img(os.path.join(output_dir, "%06d.png" % (t)))
            frame_p2 = utils.read_img(os.path.join(process_dir, "%06d.png" % (t+1)))

            ### resize image
            H_orig = frame_p2.shape[0]
            W_orig = frame_p2.shape[1]

            H_sc = int(math.ceil(float(H_orig) / args.size_multiplier) * args.size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / args.size_multiplier) * args.size_multiplier)

            frame_i1 = cv2.resize(frame_i1, (W_sc, H_sc))
            frame_i2 = cv2.resize(frame_i2, (W_sc, H_sc))
            frame_o1 = cv2.resize(frame_o1, (W_sc, H_sc))
            frame_p2 = cv2.resize(frame_p2, (W_sc, H_sc))

            with torch.no_grad():
                ### convert to tensor
                frame_i1 = utils.img2tensor(frame_i1).to(device)
                frame_i2 = utils.img2tensor(frame_i2).to(device)
                frame_o1 = utils.img2tensor(frame_o1).to(device)
                frame_p2 = utils.img2tensor(frame_p2).to(device)

                ### model input
                inputs = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)

                ### forward
                ts = time.time()

                output, lstm_state = model(inputs, lstm_state)
                frame_o2 = frame_p2 + output

                te = time.time()
                times.append(te - ts)

                ## create new variable to detach from graph and avoid memory accumulation
                lstm_state = utils.repackage_hidden(lstm_state)

                ### convert to numpy array
            frame_o2 = utils.tensor2img(frame_o2)

            ### resize to original size
            frame_o2 = cv2.resize(frame_o2, (W_orig, H_orig))

            ### save output frame
            output_filename = os.path.join(output_dir, "%06d.png" % (t+1))
            print("Saving as:", output_filename)
            utils.save_img(frame_o2, output_filename)

        ## end of frame
    ## end of video

    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" % (time_avg, len(times)))
