from image_recognition import load_model, classify_image  # Sin el prefijo 'src'

def main():
    model = load_model()
    img_path = "C:/Users/ainpo/image-project-py/src/data/560736.jpg"  # Asegúrate de la ruta correcta

    # Clasificación de la imagen
    results = classify_image(model, img_path)

    # Mostrar resultados
    print("Resultados de clasificación:")
    for label, confidence in results:
        print(f"{label}: {confidence:.2%}")

if __name__ == "__main__":
    main()
