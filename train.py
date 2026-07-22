import tensorflow as tf
from tensorflow.keras import layers, models

# 1. MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. 고정밀 연산을 위한 차원 확장 및 데이터 정규화
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# 3. 99.99% 고지를 위한 하이엔드 Deep CNN 레이어 설계
model = models.Sequential([
    # 첫 번째 컨볼루션 블록 (미세한 선 맵핑)
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(), # 연산 속도 및 안정성 극대화
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25), # 과적합 방지 차단벽

    # 두 번째 컨볼루션 블록 (글자의 꺾임과 곡선 패턴 정밀 분석)
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # 완전 연결층 (최종 분류 판단 뉴런망)
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5), # 폰트가 조금 삐뚤어져도 다 맞추도록 강건성 부여
    layers.Dense(10, activation='softmax')
])

# 4. 스마트 고주파 최적화 알고리즘 세팅
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🤖 [SYSTEM] 99.99% 정밀 조준 고성능 CNN 모델 학습을 가동합니다...")
# 인공지능이 더 정밀하게 교과서를 다독하도록 epochs를 15회로 증폭합니다.
model.fit(X_train, y_train, epochs=15, batch_size=128, validation_split=0.1, verbose=1)

# 5. 생성된 최고 사양 가중치 모델 파일 저장
model.save('mnist_cnn_model.keras')
print("🎉 [SUCCESS] 최고 사양 'mnist_cnn_model.keras' 모델 부품 생성이 완료되었습니다!")