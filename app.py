import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Couleur de fond */
    }
    .stHeader {
        background-color: #4682B4; /* Couleur de l'en-tête */
        color: #ffffff; /* Couleur du texte de l'en-tête */
        padding: 20px;
        text-align: center;
    }
    .stTitle {
        font-size: 3em;
        font-weight: bold;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton {
        background-color: #008CBA; /* Couleur du bouton */
        color: #005F6B;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="stHeader"><h1 class="stTitle">Classification d\'images sportives</h1></div>', unsafe_allow_html=True)


# Charger le modèle
class CustomResNet50(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.base_model.fc.in_features, 3) # Assuming 3 classes
        )

    def forward(self, x):
        return self.base_model(x)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model = CustomResNet50().to(DEVICE)
model.load_state_dict(torch.load("./best_model.pth", map_location=DEVICE))
model.eval()

# Charger les noms de classes
class_names = joblib.load("class_names.pkl")

# Transformations pour l'image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Interface Streamlit
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image téléversée', use_container_width=True)
    
    # Bouton "Prédire"
    if st.button("Prédire"):
        # Prédiction
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()

        # Affichage des résultats
        st.write(f"**Classe prédite : {class_names[predicted_class]} ({probabilities[predicted_class].item()*100:.2f}%)**") 
        
        # Affichage des autres classes et de leurs probabilités
        for i, probability in enumerate(probabilities):
            if i != predicted_class:
                st.write(f"{class_names[i]}: {probability.item()*100:.2f}%")
