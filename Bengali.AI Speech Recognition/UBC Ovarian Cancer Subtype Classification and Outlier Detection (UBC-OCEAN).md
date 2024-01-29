# kaggle竞赛UBC 卵巢癌亚型分类和异常值检测 （UBC-OCEAN）比赛 惜败

**比赛链接** 

https://www.kaggle.com/competitions/UBC-OCEAN

## 一、比赛背景


**比赛目标**

UBC 卵巢癌亚型 E clAssification 和异常值检测 （UBC-OCEAN） 竞赛的目标是对卵巢癌亚型进行分类。您将构建一个模型，该模型基于从 20 多个医疗中心获得的组织病理学图像的世界上最广泛的卵巢癌数据集。

您的工作将有助于提高准确卵巢癌诊断的适用性和可及性。

**上下文**

卵巢癌是女性生殖系统最致命的癌症。卵巢癌有五种常见亚型：高级别浆液性癌、透明细胞卵巢癌、子宫内膜样癌、低级别浆液性癌和粘液性癌。此外，还有几种罕见的亚型（“异常值”）。这些都具有不同的细胞形态、病因、分子和遗传特征以及临床属性。亚型特异性治疗方法越来越受到重视，尽管首先需要亚型识别，这一过程可以通过数据科学进行改进。

目前，卵巢癌的诊断依赖于病理学家来评估亚型。然而，这带来了一些挑战，包括观察者之间的分歧和诊断的可重复性。此外，服务不足的社区往往无法获得专业病理学家的帮助，即使是发达的社区也缺乏具有妇科恶性肿瘤专业知识的病理学家。

深度学习模型在分析组织病理学图像方面表现出了非凡的熟练程度。然而，挑战仍然存在，例如需要大量的训练数据，最好是来自单一来源。技术、道德和财务方面的限制，以及保密问题，使培训成为一项挑战。在本次比赛中，您将有机会获得来自四大洲 20 多个中心的最广泛和最多样化的组织病理学图像数据集。

竞赛主办方不列颠哥伦比亚大学 （UBC） 是全球教学、学习和研究中心，一直位居世界前 20 名公立大学之列。UBC拥抱创新，并将想法转化为行动。自 1915 年以来，UBC 一直为具有好奇心、动力和远见的人们打开机会之门，以塑造一个更美好的世界。加入UBC的是BC癌症，它是省卫生服务局及其世界知名的卵巢癌研究（OVCARE）团队的一部分，他们的发现导致了渐进的预防策略和改进的诊断和治疗。BC Cancer与地区卫生当局合作，为不列颠哥伦比亚省人民提供全面的癌症控制计划。我们还与卵巢肿瘤组织分析 （OTTA） 联盟合作，这是一个由来自全球超过 65 个国际团队的研究人员组成的国际多学科网络。最后，道明银行集团通过卑诗省癌症基金会（BC Cancer Foundation）慷慨捐赠，使OCEAN挑战成为可能。

您的工作可以提高识别卵巢癌亚型的准确性。更好的分类将使临床医生能够制定个性化的治疗策略，而不受地理位置的影响。这种有针对性的方法有可能提高治疗效果，减少不良反应，并最终为那些被诊断患有这种致命癌症的人带来更好的患者预后。

## 二、评估指标

使[用平衡的准确性](https://www.kaggle.com/code/metric/balanced-accuracy-score)对提交的内容进行评估。

提交文件
对于测试集中的每个变量，必须预测变量的类。该文件应包含标头，并采用以下格式：image_id label
## 三、数据集

**数据描述**

数据集描述
在这次比赛中，你面临的挑战是根据活检样本的显微镜扫描对卵巢癌的类型进行分类。

本次比赛使用隐藏测试。对提交的笔记本进行评分后，实际测试数据（包括全长示例提交）将提供给笔记本。由于数据集的大小，训练图像将不可用于提交笔记本。


**文件和字段说明**
[train/test]_images包含相关图像的文件夹。图像分为两类：全玻片图像 （WSI） 和组织微阵列 （TMA）。整个幻灯片图像的放大倍率为 20 倍，并且可能非常大。TMA 较小（大约 4,000x4,000 像素），但放大倍率为 40 倍。
测试集包含来自与火车集不同的源医院的图像，最大面积的图像接近 100,000 x 50,000 像素。我们强烈建议采用广泛的方法来考虑错误处理应管理的方案，包括图像尺寸、质量、玻片染色技术等方面的差异。预计测试集中大约有 2,000 张图像，其中大部分是 TMA。总大小为 550 GB，因此仅加载数据将非常耗时。请注意，该测试集是专门为评估模型的泛化程度而构建的。

[train/test].csv火车集的标签。
 - image_id- 每张图片都有唯一的 ID 代码。
 - label- 目标类。卵巢癌的以下亚型之一：。该类未出现在训练集中;识别异常值是本次比赛的挑战之一。仅适用于火车组。CC, EC, HGSC, LGSC, MC, Other
 - image_width - 图像宽度（以像素为单位）。
 - image_height - 图像高度（以像素为单位）。
 - is_tma - True如果载玻片是组织微阵列。仅适用于火车组。

## 四、比赛思路与实现

**模型选择**

我们最开始使用的分类模型 是vit，使用的训练数据是比赛方公布的538条训练集，训练代码和推理代码都是使用的比赛方公布的入门笔记本， 第一次提交的分数是 cv 0.38 LB 0.38。

在后面比赛只剩两周的时候我们发现讨论区私榜第二的参赛者用的是多元实例学习方法，分数都普遍的高。

最终我们选择的方式也转向了使用多元实例学习方法所构建的模型，有CLAM、ACMIL、MMIL、TransMIL。

**比赛思路**
数据处理代码 https://www.kaggle.com/code/orzlala/generate-tumor-image-patch

训练代码 https://www.kaggle.com/code/orzlala/acmil-training
         https://www.kaggle.com/code/orzlala/mmil-training
         https://www.kaggle.com/code/orzlala/transmil-training
         https://github.com/mahmoodlab/CLAM

推理代码 https://www.kaggle.com/code/orzlala/transmil-inference?scriptVersionId=156783309
         https://www.kaggle.com/code/tjl2333/cancer-subtype-lit-torch-inference-thumbnails?scriptVersionId=154527999
         
比赛思路 将整个 WSI 裁剪成数千个补丁，使用特征提取器(带Imagenet权重的resnet50)提取数据特征训练 MIL 模型。

### 训练的参数

**环境配置为**

    GPU 40G显存A100
    CUDA 11.7
    内存 64G
    python 3.8.5
    
    使用该代码训练模型最低GPU显存为32G


**训练参数**
```python
CONFIG = {
    # Config
    "config": 'UBC',
    
    # Generate
    "seed": 42,
    "devices": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "benchmark": True,
    "precision": 32,
    "epochs": 100,
    "grad_acc": 2,
    "patience": 5,
    "server": "train", #train #test
    "log_path": "logs/",
    "save_top_k": 1,
    "test_size": 0.2,
    
    # Data
    "dataset_name": "UBC_data",
    "data_shuffle": False,
    "data_dir": "/kaggle/input/feature-scale0-25/feature-0.25scale/pt_files",
    "dataset_dir": "/kaggle/working/dataset_csv/UBC16",
    "fold": 0,
    "nfold": 4,

    "train_dataloader": {
           "batch_size": 1, 
            "num_workers": 8,
       },

    "test_dataloader": {
            "batch_size": 1,
            "num_workers": 8,
        },
    
    # Model
    "name": "ACMIL",
    "n_classes": 5,
    "n_token": 5,   # 分支数量
    "n_masked_patch": 10,   # 选择前 top n_masked_patch 个 patch 的特征
    "mask_drop": 0.6,   # mask的概率
    "D_feat": 384,    # 输入特征维度
    "D_inner": 128,   # 中间特征维度
    
    # Optimizer
    "opt": "lookahead_radam",
    "lr": 0.001,
    "opt_eps": None, 
    "opt_betas": None,
    "momentum": None, 
    "weight_decay": 0.00001,
    
    # Loss
    "base_loss": "focal",
}


    trainer = Trainer(
            num_sanity_val_steps = 0,
            accelerator = CONFIG['device'],
            devices = CONFIG['devices'],
            logger = CONFIG['load_loggers'],
            callbacks = CONFIG['callbacks'],
            max_epochs = CONFIG['epochs'], 
            precision = CONFIG['precision'],  
            accumulate_grad_batches = CONFIG['grad_acc'],
#             fast_dev_run=4,
#             limit_train_batches=0.1,
#             limit_val_batches=0.1,
        )
```


**数据处理**
定义一个extract_image_tiles函数并使用Parallel并行处理wsl数据
```python
def extract_image_tiles(
    p_img, folder, size: int = 1024, stride: int = 2, rescale: float = 0.3, scale=0.25,
    drop_thr: float = 0.85, inds = None
) -> list:
    name, _ = os.path.splitext(os.path.basename(p_img))
    im = get_blend_img(int(name))
    w = h = size
    if not inds:
        # https://stackoverflow.com/a/47581978/4521646
        inds = [(y, y + h, x, x + w)
                for y in range(0, im.height, h//stride)
                for x in range(0, im.width, w//stride)]
    files, idxs, k = [], [], 0
    for idx in inds:
        y, y_, x, x_ = idx
        # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
        if drop_thr is not None:
            mask_bg = (np.sum(tile, axis=2) <= 10) | (np.max(tile, axis=2) >= 230)
            if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
                #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
                continue
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
            tile = np.zeros(tile_size, dtype=tile.dtype)
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        p_img = os.path.join(folder, f"{k:05}_{int(x_ / w)}-{int(y_ / h)}.png")
        # print(tile.shape, tile.dtype, tile.min(), tile.max())
        new_size = int(size*scale), int(size*scale)
        Image.fromarray(tile.astype(np.uint8)).resize(new_size, Image.BICUBIC).save(p_img)
        files.append(p_img)
        idxs.append(idx)
        k += 1
    return files, idxs
def extract_tiles_masks(
    idx_name,
    folder_img: str,
    size: int = 1024, stride: int = 2, rescale: float = 0.3, scale: float = 0.25,
    drop_thr: float = 0.9, drop_annos_ratio: float = 0.1
) -> None:
    idx, name = idx_name[0], str(idx_name[1])
    print(f"processing #{idx}: {name}")
    
    folder_img = os.path.join(folder_img, name)
    os.makedirs(folder_img, exist_ok=True)
    
    _, idxs = extract_image_tiles(
        os.path.join(IMAGE_FOLDER, f"{name}.png"),
        folder_img, size=size, stride=stride, rescale = rescale, scale=scale,
        drop_thr=drop_thr
    )
    rom tqdm.auto import tqdm
from joblib import Parallel, delayed

_= Parallel(n_jobs=3)(
    delayed(extract_tiles_masks)
    (id_name, size=1024, stride=2, rescale=0.3, scale=0.5, drop_thr=0.85, drop_annos_ratio=0.1, folder_img='HGSC') 
    for id_name in tqdm(enumerate(df_mask_HGSC['image_id']), total=len(df_mask_HGSC['image_id'])))

```
**训练结果**

训练代码

```python
   model_dict = {
        "name": CONFIG['name'],
        "n_classes": CONFIG['n_classes'],
    }
    model = ModelInterface(model_dict)

    trainer = Trainer(
            num_sanity_val_steps = 0,
            accelerator = CONFIG['device'],
            devices = CONFIG['devices'],
            logger = CONFIG['load_loggers'],
            callbacks = CONFIG['callbacks'],
            max_epochs = CONFIG['epochs'], 
            precision = CONFIG['precision'],  
            accumulate_grad_batches = CONFIG['grad_acc'],
#             fast_dev_run=4,
#             limit_train_batches=0.1,
#             limit_val_batches=0.1,
        )

    # 训练
    trainer.fit(model = model, datamodule = dm)
```

训练成绩

![image](https://github.com/buluslee/kaggle/assets/93359778/56f24a08-054b-4182-807e-5473fcd1992d)

看图，此图为CLAM经过10折交叉验证的训练效果图，可以看出auc指标很好，但acc由于时间匆忙，数据处理还不够精细，所以不是太理想。

因为本次比赛要求笔记本运行的时间不能超过9小时，所以我们采取了在本地训练好模型，然后上传kaggle平台，直接加载模型进行推理，这样节省了kaggle每周的gpu时间，以及代码运行的时间。

想要获得好的成绩，我们的推理代码一样很重要

### 推理部分

定义读取图像块 Patch 的 Dataset

```python
class FVDataset(Dataset):
    def __init__(self, tiles, eval_transforms = None):
        super().__init__()
        self.patch_list = tiles
        self.length = len(self.patch_list)
        if eval_transforms:
            self.transform = eval_transforms
        else:
            self.transform = transforms.Compose(
                                    [
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]
                                )
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        patch_img = self.patch_list[idx]
        patch_img = self.transform(patch_img).unsqueeze(0).to(torch.float)
        return patch_img
```
通过预训练的骨干网络推理图像的特征

```python
def collate_features(batch):
    img = torch.cat([item for item in batch], dim = 0)
    return img 


def get_slide_feature(tiles):
    """
        推理一个slide（整个图像）的所有图像块（patch）对应的特征
    """
    feature_batch_size = CONFIG['feature_batch_size']
    feature_num_workers = CONFIG['feature_num_workers']
    img_dataset = FVDataset(tiles)
    img_dataloader = DataLoader(dataset = img_dataset, batch_size = feature_batch_size, num_workers = feature_num_workers, 
                                collate_fn = collate_features)
    slide_feature = []
    for count, patch_imgs in enumerate(img_dataloader):
        with torch.no_grad():
            patch_imgs = patch_imgs.to(device)
#             print('data device:', patch_imgs.device)
            features = feature_model(patch_imgs)
            slide_feature.append(features)
            
    slide_feature = torch.cat(slide_feature, axis=0)
    return slide_feature.cpu().numpy()
```
切割整个图像为256x256的小块，并将其转化为该图的特征函数

```python
def split_subimg(
                idxs, im, w, h, size, wsi_size: int = 512, tma_size: int = 1024, 
                drop_thr: float = 0.7, white_drop_thr: float = 0.7, 
                white_thr: int = 230, 
            ):
    """
        根据idxs切割子图，可能会存在tiles筛选后为空的情况
    """
    tiles = []
    # 切割图片
    for i, idx in enumerate(idxs):
        y, y_, x, x_ = idx
        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
        
        # 如果在边缘可以填充
        if tile.shape[:2] != (h, w):
            continue
            
        if size == tma_size: 
            white_bg = np.all(tile >= white_thr, axis=2)  # 如果一个位置在三个通道上的值都大于 white_thr，那么就认为它是空白区域，删掉
            if np.sum(white_bg) >= (np.prod(white_bg.shape) * white_drop_thr):
                continue

        if size == wsi_size:
            mask_bg = (np.sum(tile, axis=2) <= 10) | (np.max(tile, axis=2) >= 230) # 对背景区域以及纯白区域的判断
            if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
                continue
        
        tile_pil = Image.fromarray(tile.astype(np.uint8))
        tiles.append(tile_pil)
    
    return tiles if tiles else None


def extract_image_tiles(
        p_img: str,
        wsi_size: int = 512, tma_size: int = 1024, scale: float =0.5, stride: int = 1,
        drop_thr: float = 0.7, white_drop_thr: float = 0.7, white_thr: int = 230, wsi_thr: int = 5500, max_samples = 2500,
    ) -> list:
    """
        输入图像路径
        输出该图像的名称image_name与特征slide_feature
    """
    im = pyvips.Image.new_from_file(p_img)
    img_name, _ = os.path.splitext(os.path.basename(p_img))
    
    if im.width < wsi_thr and im.height < wsi_thr: # 如果是TMA图片的缩略图
        size = min(tma_size, im.width, im.height)
        scale = 0.25  # 我们会对WSI图片缩放0.5，而TMA又是WSI放大了两倍的图，因此需要缩放到0.5的0.5即0.25
        stride = 2
    else: # 否则切割的大小就是0.25
        size = wsi_size 
        
    
    # 重缩放img
    im = im.resize(scale, interpolate=pyvips.Interpolate.new("bicubic"))
    w = h = int(size*scale)
    
    idxs = [(y, y + h, x, x + w)
            for y in range(0, im.height, h//stride)
            for x in range(0, im.width, w//stride)]
#     print(f'max_samples:{max_samples}, idxs:{len(idxs)}' )
    
    max_samples = max_samples if isinstance(max_samples, int) else int(len(idxs) * max_samples)
    
    if size == wsi_size and max_samples < len(idxs):
        idxs = random.sample(idxs, max_samples)
    
    tiles = split_subimg(idxs, im, w, h, size, wsi_size, tma_size, drop_thr, white_drop_thr, white_thr)
    
    if tiles is None:
        idxs = [(y, y + h, x, x + w)
            for y in range(0, im.height, h//stride)
            for x in range(0, im.width, w//stride)]
        tiles = split_subimg(idxs, im, w, h, size, wsi_size, tma_size, drop_thr, white_drop_thr, white_thr)

#     print(f'split_idxs:{len(idxs)}')
    if tiles is None:
        return img_name, None
    
    # 提取WSI图像的特征
    slide_feature = get_slide_feature(tiles)
    
    return img_name, slide_feature
```

定义MIL模型的读入数据（特征）的Dataset


```python
class UBCData(Dataset):
    def __init__(self, df_test, state=None):
        """
            给定测试数据的表格 df_test，读取表格中的图像并提取对应的特征
        """
        # Set all input args as attributes
        self.__dict__.update(locals())
        assert CONFIG['server'] == 'test', "Training stage haven't test data."
        
        #---->data and label
        self.slide_features = [extract_image_tiles(p_img = p_img, 
                                            wsi_size = CONFIG['wsi_size'], tma_size = CONFIG['tma_size'],
                                            scale = CONFIG['scale'], stride = CONFIG['stride'], 
                                            drop_thr = CONFIG['drop_thr'], white_drop_thr = CONFIG['white_drop_thr'],
                                            max_samples = CONFIG['max_samples'],
                                    )[1]      for p_img in df_test['file_path'].values ]
        
        self.length = len(self.slide_features)
        #---->order
        self.shuffle = CONFIG['data_shuffle']



    def __len__(self):
        return self.length

    
    def __getitem__(self, idx):
        feature = self.slide_features[idx]
        if feature is None:
            return torch.rand((1, 1024))
#         print('feature:', feature.shape)
        #----> shuffle
        if self.shuffle == True:
            index = [x for x in range(feature.shape[0])] # feature.shape[0] 就是当前WSI切割的块数
            random.shuffle(index)
            feature = feature[index]

        return torch.tensor(feature, dtype=torch.float) # [1, patch_num, 1024]
```
定义MIL模型：TransMIL

```python
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, **kwargs):
        # 输入特征 H
        h = kwargs['data'].float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes] 从这里这届预测出每个包在 n_classes 个类别上的特征映射结果
        Y_hat = torch.argmax(logits, dim=1) # 选择概率最大的类的下标index作为包的类别
        Y_prob = F.softmax(logits, dim = 1) # 从第1个维度开始，对每一个包的特征映射进行归一化
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat} 
        
        # 返回三个结果，特征映射的结果，归一化的概率，预测出的包的类别，通过标签训练模型
        return results_dict

```
定义通用的 Pytorch_Lightning 的模型模块 LightningModule

```python
class ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model_dict):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters() # 自动保存模型初始化的超参数
        self.load_model() # 加载对应的 MIL 模型
        self.n_classes = CONFIG['n_classes'] # MIL的输入类别


        
    def forward(self, feat):
        results_dict = self.model(data = feat)
        
        return results_dict
    
    


    def load_model(self):
        model_dict = self.hparams.model_dict
#         print(model_dict, MIL_models)
        try:
            if model_dict['model_name'] in MIL_models_dict:
                Model = MIL_models_dict[ model_dict['model_name'] ][0] # 根据这个模型的输入名称来选择相应的模型类
            else:
                raise Exception("model name is error")
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    
    def instancialize(self, Model, **other_args):
        """ 
            这里是初始化模型的输入的函数，它接受一个模型 Model，以及其他有关于模型输入的参数。
            class_args 是获得 MIL 模型 __init__ 的输入参数，由于TransMIL模型的__init__中只有 n_classes，因此 class_args 只包含 n_classes.
            self.hparams.model_dict.keys() 则是定义 LightningModule 时 __init__的输入参数中的 model_dict.
            args1 是预定义的参数，我们使用循环遍历 MIL模型初始化 __init__ 需要的参数 class_args, 如果这个参数在 LM 的初始化输入参数 model_dict 中存在
            则将其加入 args1 中，并将 args1 作为初始化实例 MIL 模型的 __init__ 参数.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model_dict.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams.model_dict[arg]
        args1.update(other_args)
        return Model(**args1)
```
模型推理

```python
def get_MIL_models(model_name, device):
    # 准备模型
    model_paths = MIL_models_dict[model_name][1]
    model_dict = {
        "model_name": model_name, # 该参数不同会产生不同的模型
        "n_classes": CONFIG['n_classes'],
    }
    model_num = len(MIL_models_dict[model_name][1])
    model_paths = MIL_models_dict[model_name][1]
    
    # 模型集成权重
    model_weights = MIL_models_dict[model_name][2]
    # 实例化模型
    model_list = [ModelInterface(model_dict).to(device) for i in range(model_num)]
    assert len(model_weights) == len(model_list)
    # 加载模型权重
    for i, model in enumerate(model_list):
        ckpt = torch.load(model_paths[i], map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

    return model_list


def get_group_MIL_models(device):
    MIL_models = {}
    MIL_model_names = MIL_models_dict.keys()
    for model_name in MIL_model_names:
        MIL_models[model_name] = get_MIL_models(model_name, device)
    return MIL_models

MIL_models = get_group_MIL_models(device)
def model_inference(model_name, MIL_dataloader, device):
    """
        对多个相同类型的模型的推理结果进行集成
    """
    model_weights = MIL_models_dict[model_name][2]
    model_list = MIL_models[model_name]
    # 对于每一个模型进行推理
    final_logits_list = []
    for feat in MIL_dataloader:
        with torch.no_grad():
            final_logits = torch.zeros(1, CONFIG['n_classes'])
            for model, weight in zip(model_list, model_weights):
                pred = model(feat.to(device))
                logits = pred['logits'].cpu()
                final_logits += logits*weight
            final_logits_list.append(final_logits)
            
    final_logits_list = torch.cat(final_logits_list)
    return final_logits_list
```
模型集成

```python
def infer_single_image(idx_row, device="cuda"):
    """
        对多个不同类型的模型的推理结果进行集成
    """
    row = idx_row[1]
    row = pd.DataFrame(row).T
    row['file_path'] = row['image_id'].apply(get_test_file_path)
    row['label'] = 0

    # 准备数据
    MIL_dataset = UBCData(row)
    MIL_dataloader = DataLoader(MIL_dataset, batch_size=CONFIG['MIL_batch_size'], num_workers=CONFIG['test_num_workers'],)
    
    # 预定义最终的结果
    final_logits_list = torch.zeros(1, CONFIG['n_classes'])
    # 不同模型之间的权重
    model_weights = [model_list[3] for model_list in MIL_models_dict.values()]
    assert len(MIL_models.keys()) == len(model_weights)
    for model_name, weight in zip(MIL_models.keys(), model_weights):
        logits_list = model_inference(model_name, MIL_dataloader, device)
        final_logits_list += (logits_list*weight)
    
    row['label'] = label_encoder[ int(torch.argmax(final_logits_list, dim=1)) ]
    return row
```

最后生成csv提交kaggle平台

```python
%%time
# scale:0.2, max_samples:200 -> 23s 
# scale:0.1, max_samples: 2000 -> 32s
# scale:0.5, max_samples: 10000 -> 53s
submission = [
    infer_single_image(idx_row, device=device)
    for idx_row in df_test.iterrows()
]
df_sub = pd.concat(submission, axis=0)
df_sub['image_id'] = df_sub['image_id'].apply(lambda img_id: int(img_id)) # 提交的submission中image_id是int类型的，label是object类型
df_sub = df_sub[["image_id", "label"]]
df_sub.to_csv("submission.csv", index=False)
```
## 五、总结
我们一共是3队参加比赛
    由于这期赛期时长较为紧迫，且运用多元实例学习方法时，赛期只剩下了两周的时间所以三队成绩都不是很理想，但后续看了私榜竞赛第二名的方案，我们和它的方式极度相似，所以，我相信我们的处理方案还是有很大的进步空间。