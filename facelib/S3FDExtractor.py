from pathlib import Path

import cv2, torch, operator
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self, n_channels):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.ones(1, self.n_channels, 1, 1))

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return self.weight * x / (norm + 1e-10)

class S3FD(nn.Module):
    def __init__(self):
        super(S3FD, self).__init__()
        
        self.minus = torch.tensor([104,117,123], dtype=torch.float32 ).view(1,3,1,1)

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.fc6 = nn.Conv2d(512, 1024, 3, 1, 3)
        self.fc7 = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.conv6_1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv6_2 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv7_1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.conv7_2 = nn.Conv2d(128, 256, 3, 2, 1)

        self.conv3_3_norm = L2Norm(256)
        self.conv4_3_norm = L2Norm(512)
        self.conv5_3_norm = L2Norm(512)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, 3, 1, 1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, 3, 1, 1)

        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, 3, 1, 1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, 3, 1, 1)

        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, 3, 1, 1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, 3, 1, 1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, 3, 1, 1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, 3, 1, 1)

        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, 3, 1, 1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, 3, 1, 1)

        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, 3, 1, 1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, 3, 1, 1)

    def forward(self, x):
        x = x.cuda()
        self.minus = self.minus.cuda()
        x = x - self.minus

        # Initial layers
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        f3_3 = self.conv3_3_norm(h)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        f4_3 = self.conv4_3_norm(h)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        f5_3 = self.conv5_3_norm(h)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        ffc7 = h 

        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        f6_2 = h

        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h))
        f7_2 = h

        cls1 = F.softmax(self.conv3_3_norm_mbox_conf(f3_3).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4), dim=-1)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        cls2 = F.softmax(self.conv4_3_norm_mbox_conf(f4_3).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2), dim=-1)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        cls3 = F.softmax(self.conv5_3_norm_mbox_conf(f5_3).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2), dim=-1)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        cls4 = F.softmax(self.fc7_mbox_conf(ffc7).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2), dim=-1)
        reg4 = self.fc7_mbox_loc(ffc7).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        cls5 = F.softmax(self.conv6_2_mbox_conf(f6_2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2), dim=-1)
        reg5 = self.conv6_2_mbox_loc(f6_2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        cls6 = F.softmax(self.conv7_2_mbox_conf(f7_2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2), dim=-1)
        reg6 = self.conv7_2_mbox_loc(f7_2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]



class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        model_path = Path(__file__).parent / "S3FD.npy"
        if not model_path.exists():
            raise Exception("Unable to load S3FD.npy")

        self.model = S3FD()
        self.device = torch.device('cuda' if torch.cuda.is_available() and not place_model_on_cpu else 'cpu')
        self.model.to(self.device)
        
        def can_squeeze(t):
            shape_set = set(t.shape)
            if len(shape_set) == 2 and 1 in shape_set:
                return True
            return False

        def reshape_tensor(k, t):
            if k == 'fc7.weight':
                return t.reshape([1024, 1024, 1, 1])
            if k in ['conv4_3_norm.weight', 'conv5_3_norm.weight']:
                return t.reshape([1, 512, 1, 1])
            if k == 'conv3_3_norm.weight':
                return t.reshape([1, 256, 1, 1])
            if can_squeeze(t):
                return t.squeeze()
            else:
                return torch.permute(t, [3, 2, 1, 0])

        def rename_key(key):
            new_key = key.split(":")[0] # discard :0 and similar
            key_elements = new_key.split("/")
            new_key = ".".join(key_elements) # replace every / with .
            return new_key

        # Load the state dictionary from the numpy file
        state_dict_np = np.load(model_path, allow_pickle=True)

        # Convert numpy arrays within the state dictionary to PyTorch tensors
        state_dict_torch = {rename_key(k): reshape_tensor(rename_key(k), torch.tensor(v, dtype=torch.float32)).cpu() for k, v in state_dict_np.items()}

        # Load the converted state dictionary into the model
        self.model.load_state_dict(state_dict_torch)


    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False  # pass exception between __enter__ and __exit__ to outer level

    def extract(self, input_image, is_bgr=True, is_remove_intersects=False):
        # Convert BGR to RGB
        if is_bgr:
            input_image = input_image[:, :, ::-1]

        (h, w, ch) = input_image.shape

        # Scale logic
        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize(input_image, (int(w/input_scale), int(h/input_scale)), interpolation=cv2.INTER_LINEAR)

        # Convert to PyTorch tensor and normalize
        input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass the tensor through the model
        outputs = self.model(input_tensor)

        detected_faces = []
        for output in outputs:
            for ltrb in output:
                print('ltrb: ', ltrb)
                l, t, r, b = [x*input_scale for x in ltrb] # <- Line causing the current issue
                if min(r-l, b-t) < 40:
                    continue
                detected_faces.append([int(x) for x in (l, t, r, b)])

        # Sort by largest area first
        detected_faces = [[(l, t, r, b), (r-l)*(b-t)] for (l, t, r, b) in detected_faces]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True)
        detected_faces = [x[0] for x in detected_faces]

        # Handling intersections
        if is_remove_intersects:
            for i in range(len(detected_faces)-1, 0, -1):
                l1, t1, r1, b1 = detected_faces[i]
                l0, t0, r0, b0 = detected_faces[i-1]
                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx >= 0) and (dy >= 0):
                    detected_faces.pop(i)

        return detected_faces


    def refine(self, olist):
        """
        Refine the output list to remove duplicates, apply confidence thresholding,
        and convert box coordinates if necessary.
        
        :param olist: A list containing raw model outputs.
        :return: A list of refined bounding boxes.
        """
        
        # Assuming each output in olist is a tuple of (confidence, bbox)
        confidences, boxes = zip(*olist)
        
        # Convert confidences and boxes to tensors for processing
        confidences = torch.tensor(confidences)
        boxes = torch.tensor(boxes)
        
        # Apply confidence thresholding (e.g., 0.5)
        mask = confidences > 0.5
        confidences = confidences[mask]
        boxes = boxes[mask]
        
        # Perform Non-Maximum Suppression
        # Here we're assuming you have an NMS function available.
        # PyTorch's torchvision library provides one such function.
        keep = nms(boxes, confidences, iou_threshold=0.5)
        
        # Convert boxes from relative to absolute coordinates if necessary
        # This step depends on how your model outputs bounding box coordinates.
        # If it's already in the desired format, you can skip this step.
        
        # width and height are assumed to be the dimensions of the input image.
        # If they are different for each image, you will need to pass them to this function.
        width, height = 300, 300  # Placeholder values
        boxes[keep, 0] *= width
        boxes[keep, 1] *= height
        boxes[keep, 2] *= width
        boxes[keep, 3] *= height
        
        return boxes[keep].tolist()

