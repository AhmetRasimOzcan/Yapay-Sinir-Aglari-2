% MNIST veri kümesini yükleme
[XTrain, YTrain] = digitTrain4DArrayData; % Eğitim verilerini ve etiketlerini yükleme

% Örneklemleri görselleştirme
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(XTrain(:,:,:,i)); % Eğitim verilerini görselleştirme
    title(YTrain(i)) % Doğrudan etiketi kullanarak başlığı ayarlama
end

% Sinir ağı modelini oluşturma
layers = [
    imageInputLayer([28 28 1]) % Giriş katmanı: 28x28x1 boyutunda bir görüntü
    convolution2dLayer(3,8,'Padding','same') % 3x3 boyutunda 8 filtreli konvolüsyon katmanı
    batchNormalizationLayer % Toplu normalizasyon katmanı
    reluLayer % ReLU aktivasyon fonksiyonu katmanı
    maxPooling2dLayer(2,'Stride',2) % 2x2 boyutunda maksimum havuzlama katmanı
    convolution2dLayer(3,16,'Padding','same') % 3x3 boyutunda 16 filtreli konvolüsyon katmanı
    batchNormalizationLayer % Toplu normalizasyon katmanı
    reluLayer % ReLU aktivasyon fonksiyonu katmanı
    maxPooling2dLayer(2,'Stride',2) % 2x2 boyutunda maksimum havuzlama katmanı
    fullyConnectedLayer(10) % Tam bağlantılı katman: 10 nöronlu
    softmaxLayer % Softmax aktivasyon fonksiyonu katmanı
    classificationLayer]; % Sınıflandırma katmanı

% Eğitim seçeneklerini tanımlama
options = trainingOptions('sgdm', ... % Stokastik gradyan inişini kullanarak eğitim seçeneklerini tanımlama
    'MaxEpochs',20, ... % Maksimum eğitim epoch sayısı
    'InitialLearnRate',0.01, ... % Başlangıç öğrenme oranı
    'Shuffle','every-epoch', ... % Her epoch'ta verileri karıştırma
    'Verbose',false, ... % Eğitim ilerlemesini konsola yazdırmama
    'Plots','training-progress'); % Eğitim ilerlemesini görselleştirme

% Sinir ağını eğitme
net = trainNetwork(XTrain,YTrain,layers,options); % Sinir ağını eğitme

% Test veri kümesini yükleme
[XTest, YTest] = digitTest4DArrayData; % Test verilerini ve etiketlerini yükleme

% Sinir ağı modelini kullanarak tahmin etme
YPred = classify(net, XTest); % Sinir ağı modelini kullanarak test verilerini sınıflandırma

% Doğruluk hesaplama
accuracy = sum(YPred == YTest) / numel(YTest); % Doğruluk hesaplama
fprintf('Test doğruluğu: %.2f%%\n', accuracy * 100); % Doğruluk sonucunu konsola yazdırma


