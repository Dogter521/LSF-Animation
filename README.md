## **LSF-Animation**
Official PyTorch implementation for the paper:

> **LSF-Animation: Label-Free Speech-Driven Facial Animation via Implicit Feature Representation. (Accepted at [ACM SIGGRAPH Asia 2025](https://asia.siggraph.org/2025))**



> We propose LSF-Animation, a novel framework that eliminates the reliance on explicit emotion and identity feature representations. Specifically, LSF-Animation implicitly extracts emotion information from speech and captures the identity features from a neutral facial mesh, enabling improved generalization to unseen speakers and emotional states without requiring manual labels. Furthermore, we introduce a Hierarchical Interaction Fusion Block (HIFB), which employs a fusion token to integrate dual transformer features and more effectively integrate emotional, motion-related and identity-related cues.

> **Note**: This project is built upon the [ProbTalk3D](https://github.com/uuembodiedsocialai/probtalk3d) framework, extending it with emotion2vec integration and improved feature representation capabilities.

### Key Features
- **Label-Free Approach**: No manual emotion or identity labels required during training
- **Universal Speech Emotion Representation**: Integration with [emotion2vec](https://huggingface.co/emotion2vec/emotion2vec_base) for robust emotion feature extraction
- **Improved Generalization**: Better performance on unseen speakers and emotional states
- **Hierarchical Feature Fusion**: Advanced HIFB architecture for effective multi-modal integration

## **TODO List**
- [x] Open-source training and testing code 
- [x] Environment setup, data processing, and pre-trained models 
- [x] Synchronous comparison experiments 
- [ ] More test results on additional datasets 


## **Framework Overview**

The overview of our LSF-Animation framework:

![LSF-Animation Pipeline](static/images/pipline.png)

Our framework consists of three main components:
1. **Audio Feature Extraction**: Extracts both acoustic and emotional features from speech using HuBERT and emotion2vec
2. **Identity Feature Extraction**: Captures identity information from neutral facial mesh parameters
3. **Hierarchical Interaction Fusion Block (HIFB)**: Integrates multi-modal features through fusion tokens and cross-attention mechanisms

### Hierarchical Interaction Fusion Block (HIFB)

One of the core innovation of our framework is the HIFB architecture, which effectively integrates multi-modal features:

![HIFB Architecture](static/images/HIFB.png)

The HIFB employs fusion tokens to query and integrate features from both audio and emotion streams through cross-attention mechanisms, enabling more effective multi-modal feature fusion for improved facial animation generation.

### Results and Comparison

![Comparison Results](static/images/comparation.png)

![User Study Results](static/images/UserStudy.png)

### Qualitative Results

We provide video demonstrations showcasing the quality of our LSF-Animation framework:

**Sequential Comparison**: [sample.mp4](static/videos/sample.mp4) - Shows results from different methods played sequentially for easy comparison.

**Synchronous Comparison Experiments**: [sample_1.mp4](static/videos/sample_1.mp4) - Shows results from different methods played simultaneously for direct visual comparison.

## **Environment**
<details><summary>Click to expand</summary>

### System Requirement
- Linux and Windows (tested on Windows 10)
- Python 3.9+
- PyTorch 2.1.1
- CUDA 12.1 (GPU with at least 2.55GB VRAM)

### Virtual Environment
```
conda create --name LSF-Animation python=3.9
conda activate LSF-Animation
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, navigate to the project `root` folder and execute:

```
pip install -r requirements.txt
```

### Additional Dependencies for Emotion2Vec
If you plan to use the emotion2vec feature extractor, install the following additional dependencies:

```bash
pip install -U funasr modelscope
# or alternatively
pip install huggingface_hub
```
</details>

## **Dataset**
<details><summary>Click to expand</summary>

Download 3DMEAD dataset following the instruction of [EMOTE](https://github.com/radekd91/inferno/tree/release/EMOTE/inferno_apps/TalkingHead/data_processing). This dataset represents facial animations using FLAME parameters.

### Data Download and Preprocess 
- Please refer to the `README.md` file in `datasets/3DMEAD_preprocess/` folder. 
- After processing, the resulting `*.npy` files will be located in the `datasets/mead/param` folder, and the `.wav` files should be in the `datasets/mead/wav` folder.

### Emotion2Vec Model Download
- Download the emotion2vec pre-trained model from [Hugging Face](https://huggingface.co/emotion2vec/emotion2vec_base) and place it in the `framework/model/feature_extractor/pretrained/emotion2vec_base/` directory.
- You can download the model using the following methods:
  - **Using Git LFS**: 
    ```bash
    cd framework/model/feature_extractor/pretrained/
    git lfs install
    git clone https://huggingface.co/emotion2vec/emotion2vec_base
    ```
  - **Using Python (huggingface_hub)**:
    ```python
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id='emotion2vec/emotion2vec_base',
        local_dir='framework/model/feature_extractor/pretrained/emotion2vec_base',
        local_dir_use_symlinks=False
    )
    ```

- <b> Optional Operation </b>
    <details><summary>Click to expand</summary>
    
    For training the comparison model in vertex space, we provide a script to transfer the FLAME parameters to vertices. Execute the script `pre_process/param_to_vert.py`. The resulting `*.npy` files should be located in the `datasets/mead/vertex` folder.
    </details>
</details>


## **Model Training**
<details><summary>Click to expand</summary>
To train the model from scratch, follow the 2-stage training approach outlined below.

### Stage 1
For the first stage of training, use the following commands:
- On Windows and Linux:
    ```
    python train_all.py experiment=vqvae_prior state=new data=mead_prior model=model_vqvae_prior
    ```
- If the Linux system has Slurm Workload Manager, use the following command: 
    ```
    sbatch train_vqvae_prior.sh
    ```

- <b> Optional Operation </b>
    <details><summary>Click to expand</summary>

    - We use Hydra configuration, which allows us to easily override settings at runtime. For example, to change the GPU ID to 1 on a multi-GPU system, set `trainer.devices=[1]`. To load a small amount of data for debugging, set `data.debug=true`.
    - To resume training from a checkpoint, set the `state` to resume and specify the `folder` and `version`. Specifically, replace the `folder` and `version` in the command below with the folder name where the checkpoint is saved. Our program generates a random name for each run, and the version is assigned automatically by the program, which may vary depending on the operating system.
        ```
        python train_all.py experiment=vqvae_prior state=resume data=mead_prior model=model_vqvae_prior folder=outputs/MEAD/vqvae_prior/XXX version=0
        ```
- <b> VAE variant </b>
    <details><summary>Click to expand</summary>  
  
    To train the VAE variant for comparison, follow the same instructions as above and change the `model` setting as below:
    ```
    python train_all.py experiment=vae_prior state=new data=mead_prior model=model_vae_prior
    ```
    </details>

### Stage 2
After completing stage 1 training, execute the following command to proceed with stage 2 training. Set `model.folder` and `model.version` to the location where the motion prior checkpoint is stored:
- On Windows and Linux:
    ```
    python train_all.py experiment=vqvae_pred state=new data=mead_pred model=model_vqvae_pred model.folder_prior=outputs/MEAD/vqvae_prior/XXX model.version_prior=0
    ```
- If the Linux system has Slurm Workload Manager, use the following command. Remember to revise the `model.folder_prior` and `model.version_prior` in the file. 
    ```
    sbatch train_vqvae_pred.sh
    ```
- <b> Optional Operation </b>
    <details><summary>Click to expand</summary>
  
    - Similar to the first stage of training, the GPU ID can be changed by setting `trainer.devices=[1]`, and debug mode can be enabled by setting `data.debug=true`.
    - To resume training from a checkpoint, set the state to `resume` and specify the `folder` and `version`: 
        ```
        python train_all.py experiment=vqvae_pred state=resume data=mead_pred model=model_vqvae_pred folder=outputs/MEAD/vqvae_pred/XXX version=0 model.folder_prior=outputs/MEAD/vqvae_prior/XXX model.version_prior=0
        ```
    </details>
- <b> VAE variant </b>
    <details><summary>Click to expand</summary>
  
    To train the VAE variant for comparison, follow the same instructions as above and change the `model` setting as below:
    ```
    python train_all.py experiment=vae_pred state=new data=mead_pred model=model_vae_pred model.folder_prior=outputs/MEAD/vae_prior/XXX model.version_prior=0
    ```
    </details>
</details>


## **Evaluation**
<details><summary>Click to expand</summary>

Download the trained model weights from [HERE](https://drive.google.com/drive/folders/1pOLZXQ7sPPf0NP7HuuW7KSf_k84gFZL1?usp=drive_link) and unzip them into the project `root` folder.

### Quantitative Evaluation
We provide code to compute the evaluation metrics mentioned in our paper. To evaluate our trained model, run the following:
- On Windows and Linux:
    ```
    python evaluation.py folder=model_weights/ProbTalk3D/stage_2 number_of_samples=10
    ```
- If the Linux system has Slurm Workload Manager, use the following command:
    ```
    sbatch evaluation.sh
    ```
- <b> Optional Operation </b>
  <details><summary>Click to expand</summary>
  
  - Adjust the GPU ID if necessary; for instance, set `device=1`.
  - To evaluate your own trained model, specify the `folder` and `version` according to the location where the checkpoint is saved:
    ```
    python evaluation.py folder=outputs/MEAD/vqvae_pred/XXX version=0 number_of_samples=10
    ```
  </details>

- <b> VAE variant </b>
    <details><summary>Click to expand</summary>
  
    To evaluate the trained VAE variant, execute the following command:
    ```
    python evaluation.py folder=model_weights/VAE_variant/stage_2 number_of_samples=10
    ```
    </details>

### Qualitative Evaluation
For qualitative evaluation, refer to the script `evaluation_quality.py`.

</details>


### Render
The generated `.npy` files contain FLAME parameters and can be rendered into videos following the below instructions. 
- We use blender to render the predicted motion. First, download the dependencies from [HERE](https://drive.google.com/file/d/1EJ0enL27YbybzUAQ3olFGhkNpEfiaoU2/view?usp=sharing) and extract them into the `deps` folder. Please note that this command can only be executed on Windows:
  ```
  python render_param.py result_folder=results/generation/vqvae_pred/stage_2/0.2 audio_folder=results/generation/test_audio
  ```
- <b> Optional Operation </b>
  <details><summary>Click to expand</summary>
  
  - To play with your own data, modify `result_folder` to where the generated `.npy` files are stored, and `audio_folder` to where the `.wav` files are located.
  - We provide post-processing code in the `post_process` folder. To change face shapes for the predicted motion, refer to the script `change_shape_param.py`.
  - To convert predicted motion to vertex space, refer to the script `post_process/transfer_to_vert.py`. For rendering animation in vertex space, use the following command on Windows and Linux: 
    ```
    python render_vert.py result_folder=results/generation/vqvae_pred/stage_2/0.2 audio_folder=results/generation/test_audio
    ```
  </details>

- <b> VAE variant </b>
  <details><summary>Click to expand</summary>

  To render the generated animations produced by the trained VAE variant, use the following command on Windows:
  ```
  python render_param.py result_folder=results/generation/vae_pred/stage_2/20 audio_folder=results/generation/test_audio
  ```
  </details>
</details>


## **Comparison**
<details><summary>Click to expand</summary>

For comparing with the diffusion model FaceDiffuser (modified version), navigate to the `diffusion` folder.
### Model training
To train the model from scratch, execute the following command:
```
python main.py
```
### Evaluation
To quantitatively evaluate our trained FaceDiffuser model, run the following command:
```
python evaluation_facediff.py --save_path "../model_weights/FaceDiffuser" --max_epoch 50
```
### Animation Generation

#### Generate Prediction
To generate animations using our trained model, execute the following command. Modify the path and style settings as needed.
```
python predict.py --save_path "../model_weights/FaceDiffuser" --epoch 50 --subject "M009" --id "M009" --emotion 6 --intensity 1 --wav_path "../results/generation/test_audio/angry.wav"
```
#### Render
Navigate back to the project `root` folder and run the following command:
```
python render_vert.py result_folder=diffusion/results/generation audio_folder=results/generation/test_audio
```
</details>

</details>

## Citation ## 
If you find the code useful for your work, please consider starring this repository and citing it:
```
@inproceedings{lu2025lsf,
        author = {Lu, Xin and Zhuang, Chuanqing and Jin, Chenxi and Lu, Zhengda and Wang, Yiqun and Liu, Wu and Xiao, Jun},
        title = {LSF-Animation: Label-Free Speech-Driven Facial Animation via Implicit Feature Representation},
        booktitle = {Proceedings of the 2025 ACM SIGGRAPH Asia Conference},
        year = {2025},
        location = {Hong Kong, Hong Kong},
        pages = {1--12},
        url = {https://doi.org/10.1145/3757377.3763887},
        doi = {10.1145/3757377.3763887},
        publisher = {ACM},
        address = {New York, NY, USA}
        } 

```

## **Acknowledgements**

This project is built upon [ProbTalk3D](https://github.com/uuembodiedsocialai/probtalk3d) and borrows and adapts code from [Learning to Listen](https://github.com/evonneng/learning2listen), [CodeTalker](https://github.com/Doubiiu/CodeTalker), [TEMOS](https://github.com/Mathux/TEMOS), [FaceXHuBERT](https://github.com/galib360/FaceXHuBERT), [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser), and [emotion2vec](https://huggingface.co/emotion2vec/emotion2vec_base). We appreciate the authors for making their code available and facilitating future research. Additionally, we are grateful to the creators of the 3DMEAD datasets used in this project.

Any third-party packages are owned by their respective authors and must be used under their respective licenses.

## **License**
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2137-3/2025/12
