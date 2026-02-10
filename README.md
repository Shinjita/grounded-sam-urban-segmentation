# Grounded SAM for Urban Scene Semantic Segmentation

This repository demonstrates the use of **Grounded Segment Anything Model (Grounded SAM)** combined with **CLIP-based semantic filtering** to perform **category-aware semantic segmentation of urban scenes**.
The workflow focuses on identifying and quantifying key visual elements in urban environments—such as **greenery, buildings, water bodies, and infrastructure**—from single RGB images. The outputs are designed to support **urban analytics, environmental exposure assessment, and GIS-based research**.

Refer to the document explaining the methodology, model logic, and output interpretation in detail: </br>
https://github.com/Shinjita/grounded-sam-urban-segmentation/blob/ee5d3252786989ecb110b067d0e75afdb9b1db8b/Semantic%20Segmentation%20with%20Grounded%20SAM_%20Comprehensive%20Guide.pdf 

## 🔍 Why this project?

Urban researchers and planners increasingly rely on street-level and window-view imagery to understand how people visually experience cities. However, translating images into **quantitative, interpretable metrics** (e.g. % green view, % built view) remains challenging. This project shows how foundation vision models can be used to:
- Perform **semantic segmentation without manual training**
- Enforce **class prioritisation** to avoid overlapping masks
- Derive **area-based metrics** (green, grey, blue exposure)
- Produce outputs suitable for **GIS and urban health research**

## 🧠 Method Overview and Key categories include:
The workflow combines:
- **Grounded SAM** for open-vocabulary segmentation  
- **CLIP** for semantic validation and filtering  
- **Priority-based masking** to resolve overlaps  
- **Pixel-based area calculation** for quantitative outputs  
- Green elements (trees, grass, shrubs, plants)
- Built elements (buildings, houses, pavements)
- Water bodies (river)
- Sky and other contextual features

## 🛠️ Technologies Used
- Python
- Grounded SAM (Segment Anything Model)
- CLIP (Vision–Language Model)
- OpenCV
- NumPy
- PyTorch
- Supervision / Autodistill


