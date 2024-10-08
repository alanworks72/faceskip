import torch
import torch.nn as nn
import os
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model_fer import MobileNetV3  # 모델 정의
from src.dataloader_fer import loadBatches

# 모델 성능 평가 함수
def evaluate_model(model, test_batches, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_batches:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# 여러 weight 파일을 불러와 가장 좋은 성능을 가진 모델 선택
def find_best_model(result_folder, model_class, num_classes, device, test_loader):
    best_acc = 0
    best_model_path = None
    weights_folder = result_folder + 'weights/'
    ln = []

    # weights 폴더의 모든 파일 불러오기
    for weight_file in os.listdir(weights_folder):
        if weight_file.endswith('.pth'):  # .pth 파일만 불러오기
            model = model_class(num_classes=num_classes)
            model.load_state_dict(torch.load(os.path.join(weights_folder, weight_file)))
            model = model.to(device)

            # 모델 성능 평가
            accuracy = evaluate_model(model, test_loader, device)
            print(f"Model {weight_file} Accuracy: {accuracy:.4f}")
            ln.append(f"Model {weight_file} Accuracy: {accuracy:.4f}")

            # 최고의 성능을 가진 모델 저장
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_path = weight_file

    print(f"Best model is {best_model_path} with accuracy {best_acc:.4f}")
    ln.append(f"Best model is {best_model_path} with accuracy {best_acc:.4f}")

    with open(result_folder+'result.txt', 'w') as f:
        f.write('\n'.join(ln))
        f.close()

    return best_model_path

# 메인 실행 함수
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 7  # FER2013 데이터셋의 클래스 개수
    batch_size = 32

    # 테스트 데이터 로드
    test_batches = loadBatches(batch_size, is_train=False)

    # 여러 가중치 파일이 저장된 폴더
    result_folder = 'result/train_13/'
    print(f'Test for {result_folder}\n')

    # 최적 모델 선택
    best_model = find_best_model(result_folder, MobileNetV3, num_classes, device, test_batches)

if __name__ == '__main__':
    main()
