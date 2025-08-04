import os  
from pathlib import Path  
from datasets import load_dataset  
import tqdm  
import logging  

# Constants  
MAT_KEYS = ["name", "basecolor", "metallic", "normal", "roughness"]  
SUFFIX_MAP = {  
    "normal": "_2K_PNG_NormalIGL.png",  
    "basecolor": "_2K_PNG_Color.png",  
    "metallic": "_2K_PNG_Metallness.png",  
    "roughness": "_2K_PNG_Roughness.png"  
}  

# Configure logging  
logging.basicConfig(filename='material_processing.log', level=logging.INFO)  

def save_material_images(x, save_dir):  
    """Save images for a single material.  
    
    Args:  
        x (dict): Material data dictionary.  
        save_dir (Path): Directory to save the images.  
    """  
    for k in MAT_KEYS:  
        if k == "name":  
            continue  
        
        try:  
            # Construct the file path  
            filename = f"{x['name']}{SUFFIX_MAP.get(k, '_2K_JPG_Unknown.png')}"  
            file_path = save_dir / filename  
            
            # Resize and save the image  
            if k in ["normal", "basecolor", "metallic", "roughness"]:  
                x[k].resize((2048, 2048)).save(file_path)  
            else:  
                x[k].save(file_path)  
                
            logging.info(f"Saved {file_path}")  
        except Exception as e:  
            logging.error(f"Failed to save {file_path}: {e}")  

def get_matsynth_material(base_output_dir):  
    """Load the MatSynth dataset and save material images.  
    
    Args:  
        base_output_dir (str): Directory to save the processed materials.  
    """  
    try:  
        # Load the dataset  
        ds = load_dataset(  
            "/baai-cwm-1/baai_cwm_ml/public_data/objects/MatSynth",  
            streaming=True,  
        )  
        ds = ds.select_columns(MAT_KEYS)  
        ds = ds.shuffle(buffer_size=1)  

        # Process each material  
        for i, x in enumerate(tqdm.tqdm(ds['train'], desc="Processing materials")):  

            print(i)
        #     save_dir = Path(os.path.join(base_output_dir, x['name']))  
        #     save_dir.mkdir(parents=True, exist_ok=True)  
        #     save_material_images(x, save_dir)  
        
        # logging.info("Processing completed successfully.")  
        # return str(save_dir.resolve())  
    
    except Exception as e:  
        logging.error(f"Error in get_matsynth_material: {e}")  
        return None  

if __name__ == "__main__":  
    base_output_dir = "/baai-cwm-1/baai_cwm_ml/public_data/rendering_data/matsynth_processed_v2/"  
    get_matsynth_material(base_output_dir)  