import numpy as np
import torch
import pandas as pd
import os
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem
from Suplement.FE.FE_BERT.FE_Bert import gnnsmile
from Suplement.FE.FE_3D.preprocess import smiles_to_data, prepare_batch
from Suplement.FE.FE_3D.pretrain_MolMVC import EmbeddingModel

def process_dataset(train_data, feature_type, device, save_dir="extracted_features"):
    os.makedirs(save_dir, exist_ok=True)

    raw_spec = str(feature_type).strip()
    norm_spec = raw_spec.replace("BERT", "bert").replace("Bert", "bert").replace(" ", "")
    is_combo = ("+" in norm_spec)
    ALLOWED = {"1D", "2D", "3D", "bert", "fingerprint"}

    if norm_spec.lower() == "multi":
        views_in_order = ["1D", "2D", "3D", "bert"]
    elif is_combo:
        parts = [p for p in norm_spec.split("+") if p]  
        views_in_order = []
        for p in parts:
            key = p if p in {"1D", "2D", "3D"} else p.lower()
            if key not in ALLOWED:
                raise ValueError(f"Unsupported feature type in combo: {p}. "
                                 f"Allowed: {sorted(list(ALLOWED))} or 'multi' or '+' combos.")
            if key not in views_in_order:
                views_in_order.append(key)
        if not views_in_order:
            raise ValueError(f"Empty/invalid combo in feature_type: {feature_type}")
    else:
        key = norm_spec if norm_spec in {"1D", "2D", "3D"} else norm_spec.lower()
        if key not in ALLOWED and key != "multi":
            raise ValueError(f"Unknown feature type: {feature_type}")
        views_in_order = [key] if key != "multi" else ["1D", "2D", "3D", "bert"]

    drug1_ids = np.array(train_data["Drug1_ID"])
    drug2_ids = np.array(train_data["Drug2_ID"])
    all_drugs = np.unique(np.append(drug1_ids, drug2_ids))
    ftrain = list(all_drugs)

    alldrugtrain = {}
    removelist = []

    print(f"Starting to process {len(ftrain)} drugs...")
    print(f"Feature spec: {raw_spec} -> views: {views_in_order}")

    for drug_id in ftrain:
        try:
            print(f"\nProcessing drug: {drug_id}")

            try:
                index = list(drug1_ids).index(drug_id)
                drug_col = "Drug1_ID"
                smiles_col = train_data.columns[1]
                print(f"Found drug in Drug1_ID column, index: {index}")
            except ValueError:
                try:
                    index = list(drug2_ids).index(drug_id)
                    drug_col = "Drug2_ID"
                    smiles_col = train_data.columns[3]
                    print(f"Found drug in Drug2_ID column, index: {index}")
                except ValueError:
                    removelist.append(drug_id)
                    print(f"Error: Drug {drug_id} not found in the dataset")
                    continue

            drug_name = train_data.iloc[index][drug_col]
            smiles = train_data.iloc[index][smiles_col]
            print(f"Drug name: {drug_name}, SMILES: {smiles}")

            need_bert = ("bert" in views_in_order)
            need_molmvc = any(v in views_in_order for v in ["1D", "2D", "3D"])
            vecs_to_concat = []
            emb_1d = None
            emb_2d = None
            emb_3d = None
            if need_molmvc:
                print(f"Using MolMVC model for {','.join([v for v in views_in_order if v in ['1D','2D','3D']])} feature extraction...")

                print("Step 1: Converting SMILES to data object...")
                try:
                    data = smiles_to_data(smiles, xyz_path=None)
                    print("Data object created successfully")
                except Exception as e:
                    print(f"Error: Unable to convert SMILES to data object: {str(e)}")
                    removelist.append(drug_id)
                    continue

                print("Step 2: Preparing batch data...")
                try:
                    batch = prepare_batch(data)
                    print("Batch data prepared successfully")
                except Exception as e:
                    print(f"Error: Unable to prepare batch data: {str(e)}")
                    removelist.append(drug_id)
                    continue

                print("Step 3: Loading EmbeddingModel...")
                try:
                    model = EmbeddingModel().to(device)
                    model.eval()
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error: Unable to load model: {str(e)}")
                    removelist.append(drug_id)
                    continue

                print("Step 4: Extracting embeddings...")
                try:
                    with torch.no_grad():
                        emb_1d_low, emb_1d_high, emb_2d_t, emb_3d_t = model(batch)
                    print("Embeddings extracted successfully")
                except Exception as e:
                    print(f"Error: Failed to extract embeddings: {str(e)}")
                    removelist.append(drug_id)
                    continue

                if "1D" in views_in_order:
                    print("Processing 1D features...")
                    emb_1d = ((emb_1d_low + emb_1d_high) / 2).cpu().numpy().flatten()
                    print(f"1D feature dimension: {emb_1d.shape[0]}")

                if "2D" in views_in_order:
                    print("Processing 2D features...")
                    emb_2d = emb_2d_t.cpu().numpy().flatten()
                    print(f"2D feature dimension: {emb_2d.shape[0]}")

                if "3D" in views_in_order:
                    print("Processing 3D features...")
                    emb_3d = emb_3d_t.cpu().numpy().flatten()
                    print(f"3D feature dimension: {emb_3d.shape[0]}")

            bert_vec = None
            if need_bert:
                print("Extracting BERT features...")
                try:
                    bert_vec = np.array(gnnsmile(smiles)[0], dtype=np.float32)
                    print(f"BERT feature dimension: {bert_vec.shape[0]}")
                except Exception as e:
                    print(f"Error: Failed to extract BERT features: {str(e)}")
                    removelist.append(drug_id)
                    continue

            for v in views_in_order:
                if v == "1D":
                    if emb_1d is None:
                        print("Error: 1D feature is required but missing.")
                        removelist.append(drug_id); break
                    vecs_to_concat.append(emb_1d)
                elif v == "2D":
                    if emb_2d is None:
                        print("Error: 2D feature is required but missing.")
                        removelist.append(drug_id); break
                    vecs_to_concat.append(emb_2d)
                elif v == "3D":
                    if emb_3d is None:
                        print("Error: 3D feature is required but missing.")
                        removelist.append(drug_id); break
                    vecs_to_concat.append(emb_3d)
                elif v == "bert":
                    if bert_vec is None:
                        print("Error: BERT feature is required but missing.")
                        removelist.append(drug_id); break
                    vecs_to_concat.append(bert_vec)
                elif v == "fingerprint":
                    print("Error: fingerprint not supported inside combos in this implementation.")
                    removelist.append(drug_id); break
                else:
                    print(f"Error: Unknown view '{v}' in combo.")
                    removelist.append(drug_id); break

            if drug_id in removelist:
                continue

            if len(views_in_order) == 1 and views_in_order[0] == "fingerprint":
                print("Using fingerprint feature extraction...")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Error: Cannot create molecule from SMILES: {smiles}")
                        removelist.append(drug_id)
                        continue
                    fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                    fp_vec = np.array([float(bit) for bit in fps.ToBitString()], dtype=np.float32)
                    vecs_to_concat = [fp_vec]
                    print(f"Fingerprint feature dimension: {fp_vec.shape[0]}")
                except Exception as e:
                    print(f"Error: Failed to extract fingerprint features: {str(e)}")
                    removelist.append(drug_id)
                    continue

            drug_fingerprints = np.concatenate(vecs_to_concat, axis=-1).astype(np.float32)
            print(f"Final feature dim ({'+'.join(views_in_order)}): {drug_fingerprints.shape[0]}")

            alldrugtrain[drug_id] = drug_fingerprints.tolist()
            print(f"Successfully extracted features for drug {drug_id}")

        except Exception as e:
            print(f"An uncaught exception occurred while processing drug {drug_id}: {str(e)}")
            traceback.print_exc()
            removelist.append(drug_id)

    for drug_id in removelist:
        if drug_id in ftrain:
            ftrain.remove(drug_id)

    save_key = ("multi" if norm_spec.lower() == "multi" else "+".join(views_in_order)).replace("+", "_")

    if alldrugtrain:
        feature_matrix = np.array(list(alldrugtrain.values()), dtype=np.float32)

        np.save(f"{save_dir}/{save_key}_features.npy", feature_matrix)
        np.save(f"{save_dir}/{save_key}_drug_ids.npy", np.array(ftrain))

        print(f"\n=== {raw_spec} features stored successfully ===")
        print(f"Number of drugs: {len(alldrugtrain)}")
        print(f"Feature dimension: {feature_matrix.shape[1]}")
        print(f"Total samples: {feature_matrix.shape[0]}")
        print(f"Storage path: {save_dir}/{save_key}_features.npy")

        if np.isnan(feature_matrix).any():
            print("Warning: Feature data contains NaN values!")
        if np.isinf(feature_matrix).any():
            print("Warning: Feature data contains Inf values!")

    else:
        print("Warning: No drug features were successfully extracted!")

    print(f"\nProcessing statistics:")
    print(f"Total number of drugs: {len(ftrain) + len(removelist)}")
    print(f"Successfully processed: {len(alldrugtrain)}")
    print(f"Failed processing: {len(removelist)}")

    if removelist:
        pd.DataFrame({'failed_drugs': removelist}).to_csv(f"{save_dir}/failed_drugs.csv", index=False)
        print(f"Saved failed drugs list to {save_dir}/failed_drugs.csv")

    return alldrugtrain, ftrain
