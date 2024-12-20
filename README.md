# DCGAN_Fashion_Image_Synthesis_project
A project utilizing DCGANs to generate realistic synthetic fashion images using the Fashion-MNIST dataset, showcasing the power of adversarial learning in image synthesis.

**Description**  
A project utilizing Deep Convolutional Generative Adversarial Networks (DCGANs) to generate realistic synthetic fashion images. The project demonstrates the application of adversarial learning in the field of fashion image synthesis, using the Fashion-MNIST dataset as the baseline.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Dataset](#dataset)
5. [Technologies Used](#technologies-used)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Future Scope](#future-scope)
9. [Contributors](#contributors)
10. [License](#license)

---

## **Introduction**

This project focuses on generating synthetic images of fashion items using Deep Convolutional Generative Adversarial Networks (DCGANs). The generator network creates images, while the discriminator network distinguishes between real and fake images, improving the quality of generated images through adversarial training.

**Key Objectives**:
- Generate high-quality synthetic images of fashion items like shoes, dresses, and shirts.
- Demonstrate the capabilities of adversarial training using GANs.
- Visualize the progression of generated images over epochs.

---

## **Project Structure**

```plaintext
DCGAN_Fashion_Image_Synthesis/
├── data/
│   ├── fashion_mnist/               # Dataset files
│
├── notebooks/
│   ├── DCGAN_Fashion_Image_Synthesis.ipynb  # Jupyter Notebook for training
│
├── models/
│   ├── generator_model.pth         # Trained generator model
│   ├── discriminator_model.pth     # Trained discriminator model
│
├── results/
│   ├── generated_images/           # Folder for generated image samples
│
├── requirements.txt                # Dependencies for the project
├── README.md                       # Project documentation
└── LICENSE                         # License file
```

---

## **Features**

- **Generative Adversarial Training**: 
  - Implemented DCGAN architecture to train generator and discriminator models adversarially.
- **Realistic Image Generation**:
  - Generated synthetic images of fashion items that resemble real data.
- **Visualization**:
  - Visualized the improvement of generated images over training epochs.
- **Reproducibility**:
  - Saved model weights for future use.

---

## **Dataset**

- **Name**: Fashion-MNIST Dataset  
- **Description**: A dataset of 28x28 grayscale images of clothing items, categorized into 10 classes (e.g., shirts, shoes, dresses).  
- **Source**: [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)

---

## **Technologies Used**

- **Programming Language**: Python  
- **Libraries**:
  - PyTorch/TensorFlow (choose based on your implementation)
  - NumPy
  - Matplotlib
  - Jupyter Notebook
- **Framework**: DCGAN (Deep Convolutional Generative Adversarial Networks)

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DCGAN_Fashion_Image_Synthesis.git
   cd DCGAN_Fashion_Image_Synthesis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/DCGAN_Fashion_Image_Synthesis.ipynb
   ```

4. Monitor training and visualize generated images in the notebook.

---

## **Results**

- **Generated Images**: Successfully synthesized realistic images of clothing items after training for **50 epochs**.
- **Visualization**: Improved image quality observed over epochs. Generated samples are saved in the `results/generated_images` directory.


---

## **Future Scope**

- **High-Resolution Images**:
  - Extend the model to generate higher-resolution images.
- **Conditional GANs**:
  - Implement conditional GANs (cGANs) to generate specific categories of fashion items.
- **New Datasets**:
  - Experiment with larger and more diverse datasets like CIFAR-10 or custom datasets.

---

## **Contributors**

- **Rishikesh** - Developer
- LinkedIn: www.linkedin.com/in/rishikesh-a12090285
- Email: rishikesh23@kgpian.iitkgp.ac.in


---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
