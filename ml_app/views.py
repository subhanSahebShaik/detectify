from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm
import datetime
from random import choice

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

index_template_name = 'index.html'
predict_template_name = 'predict.html'

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


class Model(nn.Module):

    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim,
                            lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            # if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if (len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """
        # if(len(frames)<self.count):
        #   for i in range(self.count-len(frames)):
        #         frames.append(self.transform(frame))
        # print("no of frames", self.count)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image


def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    image = image*[0.22803, 0.22145, 0.216989] + [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()


def predict(model, img, path='./', video_file_name=""):
    fmap, logits = model(img.to('cpu'))
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    print('confidence of prediction:',
          logits[:, int(prediction.item())].item()*100)
    return [int(prediction.item()), confidence]


def plot_heat_map(i, model, img, path='./', video_file_name=''):
    fmap, logits = model(img.to('cpu'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    # out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
    out = np.dot(fmap[i].detach().cpu().numpy().reshape(
        (nc, h*w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255*predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    result = heatmap * 0.5 + img*0.8*255
    # Saving heatmap - Start
    heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
    image_name = os.path.join(settings.PROJECT_DIR,
                              'uploaded_images', heatmap_name)
    cv2.imwrite(image_name, result)
    # Saving heatmap - End
    result1 = heatmap * 0.5/255 + img*0.8
    r, g, b = cv2.split(result1)
    result1 = cv2.merge((r, g, b))
    return image_name

# Model Selection


def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(
        settings.PROJECT_DIR, "models", "*.pt"))

    for i in list_models:
        model_name.append(i.split("\\")[-1])

    for i in model_name:
        try:
            seq = i.split("_")[3]
            if (int(seq) == sequence_length):
                sequence_model.append(i)
        except:
            pass

    if len(sequence_model) > 1:
        accuracy = []
        for i in sequence_model:
            acc = i.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = sequence_model[max_index]
    else:
        final_model = sequence_model[0]
    return final_model


ALLOWED_VIDEO_EXTENSIONS = set(
    ['mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'])


def allowed_video_file(filename):
    # print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else:
        return False


def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error(
                        "upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error(
                    "sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})

            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error(
                    "upload_video_file", "Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})

            saved_video_file = 'uploaded_file_' + \
                str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(
                    settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', 'app', 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(
                    settings.PROJECT_DIR, 'uploaded_videos', 'app', 'uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})


def result(request):
    preprocessed_images = request.session["preprocessed_images"]
    heatmap_images = request.session["heatmap_images"]
    faces_cropped_images = request.session["faces_cropped_images"]
    models_location = request.session["models_location"]
    output = request.session["output"]
    confidence = request.session["confidence"]
    accuracy = request.session["accuracy"]
    decimal = request.session["decimal"]
    video_file_name = request.session["original_video"]
    return render(request, predict_template_name, {'preprocessed_images': preprocessed_images, 'heatmap_images': heatmap_images, "faces_cropped_images": faces_cropped_images, "original_video": video_file_name, "models_location": models_location, "output": output, "confidence": confidence, "accuracy": accuracy, "decimal": decimal})


def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = video_file.split('\\')[-1]
        if settings.DEBUG == False:
            production_video_name = video_file_name.split('/')[3:]
            production_video_name = '/'.join([str(elem)
                                             for elem in production_video_name])
            print("Production file name", production_video_name)
        video_file_name_only = video_file_name.split('.')[0]
        video_dataset = validation_dataset(
            path_to_videos, sequence_length=sequence_length, transform=train_transforms)
        model = Model(2).cpu()

        model_name = os.path.join(
            settings.PROJECT_DIR, 'models', get_accurate_model(sequence_length))
        models_location = os.path.join(settings.PROJECT_DIR, 'models')
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(
            path_to_model, map_location=torch.device('cpu')))
        model.eval()
        start_time = time.time()
        # Start: Displaying preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)

        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()

        for i in range(1, sequence_length+1):
            frame = frames[i]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = pImage.fromarray(image, 'RGB')
            image_name = video_file_name_only+"_preprocessed_"+str(i)+'.png'
            if settings.DEBUG:
                image_path = os.path.join(
                    settings.PROJECT_DIR, 'uploaded_images', image_name)
            else:
                print("image_name", image_name)
                image_path = "/home/app/staticfiles" + image_name
            img.save(image_path)
            preprocessed_images.append(image_name)
        print("<=== | Videos Splitting Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))
        # End: Displaying preprocessing images

        # Start: Displaying Faces Cropped Images
        print("<=== | Started Face Cropping Each Frame | ===>")
        padding = 40
        faces_found = 0
        for i in range(1, sequence_length+1):
            frame = frames[i]
            # fig, ax = plt.subplots(1,1, figsize=(5, 5))
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 0:
                continue
            top, right, bottom, left = face_locations[0]
            frame_face = frame[top-padding:bottom +
                               padding, left-padding:right+padding]
            image = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

            img = pImage.fromarray(image, 'RGB')
            image_name = video_file_name_only+"_cropped_faces_"+str(i)+'.png'
            if settings.DEBUG:
                image_path = os.path.join(
                    settings.PROJECT_DIR, 'uploaded_images', video_file_name_only+"_cropped_faces_"+str(i)+'.png')
            else:
                image_path = "/home/app/staticfiles" + image_name
            img.save(image_path)
            faces_found = faces_found + 1
            faces_cropped_images.append(image_name)
        print("<=== | Face Cropping Each Frame Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face is detected
        if faces_found == 0:
            return render(request, predict_template_name, {"no_faces": True})

        # End: Displaying Faces Cropped Images
        try:
            heatmap_images = []
            for i in range(0, len(path_to_videos)):
                output = ""
                print("<=== | Started Predicition | ===>")
                prediction = predict(
                    model, video_dataset[i], './', video_file_name_only)
                confidence = round(prediction[1], 1)
                print("<=== |  Predicition Done | ===>")
                # print("<=== | Heat map creation started | ===>")
                # for j in range(0, sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))
                if prediction[0] == 1:
                    output = "REAL"
                else:
                    output = "FAKE"
                print("Prediction : ",
                      prediction[0], "==", output, "Confidence : ", confidence)
                print("--- %s seconds ---" % (time.time() - start_time))
            request.session["preprocessed_images"] = preprocessed_images
            request.session["heatmap_images"] = heatmap_images
            request.session["faces_cropped_images"] = faces_cropped_images
            request.session["models_location"] = models_location
            request.session["output"] = output
            request.session["confidence"] = confidence
            request.session["accuracy"] = model_name.split("\\")[-1][6:8]
            request.session["decimal"] = model_name.split("\\")[-1][13:15]
            if settings.DEBUG:
                request.session["original_video"] = video_file_name
                # return render(request, predict_template_name, {'preprocessed_images': preprocessed_images, 'heatmap_images': heatmap_images, "faces_cropped_images": faces_cropped_images, "original_video": video_file_name, "models_location": models_location, "output": output, "confidence": confidence, "accuracy": model_name.split("\\")[-1][6:8], "decimal": model_name.split("\\")[-1][13:15]})
            else:
                request.session["original_video"] = production_video_name
                # return render(request, predict_template_name, {'preprocessed_images': preprocessed_images, 'heatmap_images': heatmap_images, "faces_cropped_images": faces_cropped_images, "original_video": production_video_name, "models_location": models_location, "output": output, "confidence": confidence, "accuracy": model_name.split("\\")[-1][6:8], "decimal": model_name.split("\\")[-1][13:15]})
            return redirect("ml_app:result")
        except Exception as e:
            print("Execption :", e.args)
            return render(request, 'cuda_full.html')


def about(request):
    return render(request, "about.html")


def handler404(request, exception):
    return render(request, '404.html', status=404)


@csrf_exempt
def video_upload_api(request):
    if request.method == 'POST':
        file_name = f"uploaded_video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        with open(f"uploaded_videos/{file_name}", "wb") as f:
            f.write(request.body)
        request.session['file_name'] = os.path.join(
            settings.PROJECT_DIR, 'uploaded_videos', file_name)
        request.session["sequence_length"] = 20
        result = run_deep_fake_detection(request)
        return JsonResponse({'result': result})
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)


def run_deep_fake_detection(request):
    if 'file_name' in request.session:
        video_file = request.session['file_name']
    if 'sequence_length' in request.session:
        sequence_length = request.session['sequence_length']
    else:
        sequence_length = choice([20, 40, 60, 80, 100])
    path_to_videos = [video_file]
    video_file_name = video_file.split('\\')[-1]
    if settings.DEBUG == False:
        production_video_name = video_file_name.split('/')[3:]
        production_video_name = '/'.join([str(elem)
                                         for elem in production_video_name])
    video_file_name_only = video_file_name.split('.')[0]
    video_dataset = validation_dataset(
        path_to_videos, sequence_length=sequence_length, transform=train_transforms)
    model = Model(2).cpu()

    model_name = os.path.join(settings.PROJECT_DIR,
                              'models', get_accurate_model(sequence_length))
    models_location = os.path.join(settings.PROJECT_DIR, 'models')
    path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
    model.load_state_dict(torch.load(
        path_to_model, map_location=torch.device('cpu')))
    model.eval()
    start_time = time.time()
    # Start: Displaying preprocessing images
    preprocessed_images = []
    faces_cropped_images = []
    cap = cv2.VideoCapture(video_file)

    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

    for i in range(1, sequence_length+1):
        frame = frames[i]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = pImage.fromarray(image, 'RGB')
        image_name = video_file_name_only+"_preprocessed_"+str(i)+'.png'
        if settings.DEBUG:
            image_path = os.path.join(
                settings.PROJECT_DIR, 'uploaded_images', image_name)
        else:
            image_path = "/home/app/staticfiles" + image_name
        img.save(image_path)
        preprocessed_images.append(image_name)
    # End: Displaying preprocessing images
    # Start: Displaying Faces Cropped Images
    padding = 40
    faces_found = 0
    for i in range(1, sequence_length+1):
        frame = frames[i]
        # fig, ax = plt.subplots(1,1, figsize=(5, 5))
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 0:
            continue
        top, right, bottom, left = face_locations[0]
        frame_face = frame[top-padding:bottom +
                           padding, left-padding:right+padding]
        image = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

        img = pImage.fromarray(image, 'RGB')
        image_name = video_file_name_only+"_cropped_faces_"+str(i)+'.png'
        if settings.DEBUG:
            image_path = os.path.join(
                settings.PROJECT_DIR, 'uploaded_images', video_file_name_only+"_cropped_faces_"+str(i)+'.png')
        else:
            image_path = "/home/app/staticfiles" + image_name
        img.save(image_path)
        faces_found = faces_found + 1
        faces_cropped_images.append(image_name)
    # No face is detected
    if faces_found == 0:
        return render(request, predict_template_name, {"no_faces": True})

    # End: Displaying Faces Cropped Images
    try:
        for i in range(0, len(path_to_videos)):
            prediction = predict(
                model, video_dataset[i], './', video_file_name_only)
            confidence = round(prediction[1], 1)
        return {'output': 'REAL' if prediction[0] == 1 else 'FAKE', 'confidence': confidence, 'prediction': prediction[0], 'Number of frames used': sequence_length}
    except Exception as e:
        return {"Exception": e}


def get_data(request):
    is_splitted = request.session["is_video_splitted"] if "is_video_splitted" in request.session else None
    data = {"is_splitted": is_splitted}
    return JsonResponse(data)
