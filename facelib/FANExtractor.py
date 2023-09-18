import torch, cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from facelib import FaceType, LandmarksProcessor


class FANExtractor(object):
    def __init__(self, landmarks_3D=False, place_model_on_cpu=False):

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

        # def rename_key(key):
        #     new_key = key.split(":")[0] # discard :0 and similar
        #     key_elements = new_key.split("/")
        #     new_key = ".".join(key_elements) # replace every / with .
        #     return new_key
        
        model_path = Path(__file__).parent / ("2DFAN.npy" if not landmarks_3D else "3DFAN.npy")
        if not model_path.exists():
            raise Exception("Unable to load FANExtractor model")

        self.device = torch.device('cpu') if place_model_on_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class ConvBlock(nn.Module):
            def __init__(self, in_planes, out_planes):
                super(ConvBlock, self).__init__()

                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = nn.Conv2d(in_planes, out_planes // 2, kernel_size=3, stride=1, padding=1, bias=False)

                self.bn2 = nn.BatchNorm2d(out_planes // 2)
                self.conv2 = nn.Conv2d(out_planes // 2, out_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

                self.bn3 = nn.BatchNorm2d(out_planes // 4)
                self.conv3 = nn.Conv2d(out_planes // 4, out_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

                self.downsample = None
                if in_planes != out_planes:
                    self.downsample = nn.Sequential(
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(),
                        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
                    )

            def forward(self, x):
                residual = x

                out1 = F.relu(self.bn1(x))
                out1 = self.conv1(out1)

                out2 = F.relu(self.bn2(out1))
                out2 = self.conv2(out2)

                out3 = F.relu(self.bn3(out2))
                out3 = self.conv3(out3)

                out = torch.cat([out1, out2, out3], dim=1)

                if self.downsample:
                    residual = self.downsample(x)

                out += residual
                return out


        class HourGlass(nn.Module):
            def __init__(self, in_planes, depth):
                super(HourGlass, self).__init__()
                self.b1 = ConvBlock(in_planes, 256)
                self.b2 = ConvBlock(in_planes, 256)

                self.b2_plus = ConvBlock(256, 256) if depth == 1 else HourGlass(256, depth - 1)

                self.b3 = ConvBlock(256, 256)

            def forward(self, x):
                up1 = self.b1(x)

                low = F.avg_pool2d(x, kernel_size=2, stride=2)
                low1 = self.b2(low)

                low2 = self.b2_plus(low1)
                low3 = self.b3(low2)

                up2 = F.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

                return up1 + up2


        class FAN(nn.Module):
            def __init__(self):
                super(FAN, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)

                self.conv2 = ConvBlock(64, 128)
                self.conv3 = ConvBlock(128, 128)
                self.conv4 = ConvBlock(128, 256)

                self.hgs = nn.ModuleList([HourGlass(256, 4) for _ in range(4)])
                self.residuals = nn.ModuleList([ConvBlock(256, 256) for _ in range(4)])
                self.lstm1s = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=1, stride=1) for _ in range(4)])
                self.bn_end = nn.ModuleList([nn.BatchNorm2d(256) for _ in range(4)])
                self.lstm2s = nn.ModuleList([nn.Conv2d(256, 68, kernel_size=1, stride=1) for _ in range(4)])
                # self.bl = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=1, stride=1) for _ in range(3)])
                # self.al = nn.ModuleList([nn.Conv2d(68, 256, kernel_size=1, stride=1) for _ in range(3)])
                self.bl = nn.ModuleList([nn.Conv2d(68, 256, kernel_size=1, stride=1) for _ in range(3)])
                self.al = nn.ModuleList([nn.Conv2d(68, 256, kernel_size=1, stride=1) for _ in range(3)])


            def forward(self, inp):
                x, = inp

                x = F.relu(self.bn1(self.conv1(x)))
                x = self.conv2(x)
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                x = self.conv3(x)
                x = self.conv4(x)

                previous = x
                for i in range(4):
                    ll = self.hgs[i](previous)
                    ll = self.residuals[i](ll)
                    ll = F.relu(self.bn_end[i](self.lstm1s[i](ll)))
                    tmp_out = self.lstm2s[i](ll)
                    
                    if i < 3:
                        ll = ll + self.bl[i](tmp_out) + self.al[i](tmp_out)
                        previous = ll


                x = tmp_out
                x = x.permute(0, 3, 1, 2)
                return x

            def build_for_run(self, shapes_list):
                if not isinstance(shapes_list, list):
                    raise ValueError("shapes_list must be a list.")
                self.run_placeholders = [torch.zeros(*sh) for _, sh in shapes_list]
                self.run_output = self.__call__(self.run_placeholders)

        def rename_key(key):
            # Replace m_0 with hgs.0
            return key.replace("m_0", "hgs.0")

        e = None
        if place_model_on_cpu:
            device = torch.device("cpu")
            self.model.to(device)

        if e is not None: e.__enter__()
        self.model = FAN()
        # self.model.load_weights(str(model_path))
        if e is not None: e.__exit__(None,None,None)

        self.model.build_for_run ([(torch.float32, (1, 3, 256, 256))])
        
        self.model = FAN().to(self.device)

        # Load the state dictionary from the numpy file
        state_dict_np = np.load(model_path, allow_pickle=True)

        # Convert numpy arrays within the state dictionary to PyTorch tensors
        state_dict_torch = {rename_key(k): reshape_tensor(rename_key(k), torch.tensor(v, dtype=torch.float32)).cpu() for k, v in state_dict_np.items()}

        # Load the converted state dictionary into the model
        self.model.load_state_dict(state_dict_torch, strict=False)

        # missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_torch, strict=False)
        # print("Missing Keys:", missing_keys)
        # print("Unexpected Keys:", unexpected_keys)

    
    def extract(self, input_image, rects, second_pass_extractor=None, is_bgr=True, multi_sample=False):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:, :, ::-1]  # Convert to RGB

        h, w, ch = input_image.shape
        landmarks = []

        for (left, top, right, bottom) in rects:
            scale = (right - left + bottom - top) / 195.0
            center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
            centers = [center]

            if multi_sample:
                centers += [center + [-1, -1], center + [1, -1], center + [1, 1], center + [-1, 1]]

            images = []
            ptss = []

            try:
                for c in centers:
                    images.append(self.crop(input_image, c, scale))

                images = torch.stack([torch.from_numpy(img) for img in images])
                images = images.float().div_(255.0).permute(0, 3, 1, 2).to(self.device)  # Convert images to [B, C, H, W]

                with torch.no_grad():
                    predicted = self.model(images)
                
                predicted = [p.cpu().numpy() for p in predicted]

                for i, pred in enumerate(predicted):
                    ptss.append(self.get_pts_from_predict(pred, centers[i], scale))
                pts_img = np.mean(np.array(ptss), 0)

                landmarks.append(pts_img)
            except:
                landmarks.append(None)

            if second_pass_extractor is not None:
                for i, lmrks in enumerate(landmarks):
                    try:
                        if lmrks is not None:
                            image_to_face_mat = LandmarksProcessor.get_transform_mat(lmrks, 256, FaceType.FULL)
                            face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC)

                            # Assuming second_pass_extractor's extract method has a similar signature:
                            rects2 = second_pass_extractor.extract(face_image, is_bgr=False)
                            if len(rects2) == 1: 
                                lmrks2 = self.extract(face_image, [rects2[0]], is_bgr=False, multi_sample=True)[0]
                                landmarks[i] = LandmarksProcessor.transform_points(lmrks2, image_to_face_mat, True)
                    except:
                        pass

        return landmarks


    def transform(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype( np.int )
        br = self.transform([resolution, resolution], center, scale, resolution).astype( np.int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        a_ch, a_h, a_w = a.shape

        b = a.reshape(a_ch, a_h * a_w)
        c = b.argmax(axis=1).repeat(2).reshape(a_ch, 2).astype(np.float32)
        c[:, 0] %= a_w
        c[:, 1] = np.floor(c[:, 1] / a_w)

        for i in range(a_ch):
            pX, pY = int(c[i, 0]), int(c[i, 1])
            if 0 < pX < a_w - 1 and 0 < pY < a_h - 1:
                diff = np.array([a[i, pY, pX+1] - a[i, pY, pX-1], 
                                a[i, pY+1, pX] - a[i, pY-1, pX]])
                c[i] += np.sign(diff) * 0.25

        c += 0.5
        return np.array([self.transform(c[i], center, scale, a_w) for i in range(a_ch)])