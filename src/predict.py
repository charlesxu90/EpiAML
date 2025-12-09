"""
EpiAML Prediction Script

Make predictions using trained EpiAML model.
"""

import os
import argparse
import numpy as np
import torch
from model import EpiAMLModel
from data_utils import load_training_data


def predict_batch(
    model_path,
    input_file,
    output_file,
    feature_order=None,
    class_mapping=None,
    batch_size=32,
    device='cuda'
):
    """
    Make predictions on a batch of samples.

    Args:
        model_path (str): Path to trained model
        input_file (str): Path to input data (.h5 or .csv)
        output_file (str): Path to output CSV file
        feature_order (str): Path to feature order file
        class_mapping (str): Path to class mapping CSV
        batch_size (int): Batch size for prediction
        device (str): Device ('cuda' or 'cpu')

    Returns:
        dict: Prediction results
    """
    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = EpiAMLModel.load_model(model_path, device=device)
    model.eval()
    print(f"  Model loaded: {model.get_num_parameters():,} parameters")

    # Load data
    print(f"\nLoading data from {input_file}...")
    data, true_labels, feature_names = load_training_data(
        input_file,
        format='auto',
        binarize=True,
        feature_order=feature_order
    )

    print(f"  Loaded: {data.shape[0]} samples × {data.shape[1]} features")

    # Load class mapping if provided
    class_names = None
    if class_mapping and os.path.exists(class_mapping):
        try:
            import pandas as pd
            mapping_df = pd.read_csv(class_mapping)
            class_names = {int(row['class_index']): row['class_name']
                          for _, row in mapping_df.iterrows()}
            print(f"  Loaded class mapping: {len(class_names)} classes")
        except Exception as e:
            print(f"  Warning: Could not load class mapping: {e}")

    # Make predictions
    print(f"\nMaking predictions...")
    data_tensor = torch.FloatTensor(data).to(device)

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data_tensor[i:i + batch_size]
            probs = model.predict_proba(batch)

            all_probabilities.append(probs.cpu().numpy())
            predictions = probs.argmax(dim=1).cpu().numpy()
            all_predictions.append(predictions)

    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    confidences = probabilities.max(axis=1)

    print(f"  Predictions complete: {len(predictions)} samples")

    # Prepare results
    try:
        import pandas as pd

        results_dict = {
            'sample_id': range(len(predictions)),
            'predicted_class': predictions,
            'confidence': confidences
        }

        # Add predicted class names if mapping available
        if class_names:
            results_dict['predicted_class_name'] = [
                class_names.get(p, f'Class_{p}') for p in predictions
            ]

        # Add true labels if available
        if true_labels is not None:
            results_dict['true_label'] = true_labels
            results_dict['correct'] = (predictions == true_labels).astype(int)

            # Calculate accuracy
            accuracy = (predictions == true_labels).mean() * 100
            print(f"  Accuracy: {accuracy:.2f}%")

        # Add probability for each class
        for i in range(probabilities.shape[1]):
            class_name = class_names[i] if class_names else f'class_{i}'
            results_dict[f'prob_{class_name}'] = probabilities[:, i]

        # Save to CSV
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences,
            'accuracy': accuracy if true_labels is not None else None
        }

    except ImportError:
        print("Warning: pandas not available, saving as numpy array")
        np.save(output_file.replace('.csv', '_predictions.npy'), predictions)
        np.save(output_file.replace('.csv', '_probabilities.npy'), probabilities)
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences
        }


def predict_single(
    model_path,
    input_file,
    class_mapping=None,
    feature_order=None,
    device='cuda'
):
    """
    Make prediction on a single sample.

    Args:
        model_path (str): Path to trained model
        input_file (str): Path to input data file (single sample)
        class_mapping (str): Path to class mapping CSV
        feature_order (str): Path to feature order file
        device (str): Device ('cuda' or 'cpu')

    Returns:
        dict: Prediction result
    """
    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = EpiAMLModel.load_model(model_path, device=device)
    model.eval()

    # Load data
    data, _, _ = load_training_data(
        input_file,
        format='auto',
        binarize=True,
        feature_order=feature_order
    )

    if len(data) != 1:
        print(f"Warning: Input file contains {len(data)} samples, using only the first one")
        data = data[:1]

    # Load class mapping
    class_names = None
    if class_mapping and os.path.exists(class_mapping):
        try:
            import pandas as pd
            mapping_df = pd.read_csv(class_mapping)
            class_names = {int(row['class_index']): row['class_name']
                          for _, row in mapping_df.iterrows()}
        except:
            pass

    # Make prediction
    data_tensor = torch.FloatTensor(data).to(device)

    with torch.no_grad():
        probs = model.predict_proba(data_tensor)

    probs = probs.cpu().numpy()[0]
    prediction = probs.argmax()
    confidence = probs[prediction]

    # Get top 5 predictions
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_predictions = []
    for idx in top5_indices:
        class_name = class_names[idx] if class_names else f'Class_{idx}'
        top5_predictions.append({
            'class_index': int(idx),
            'class_name': class_name,
            'probability': float(probs[idx])
        })

    result = {
        'predicted_class': int(prediction),
        'predicted_class_name': class_names[prediction] if class_names else f'Class_{prediction}',
        'confidence': float(confidence),
        'top_predictions': top5_predictions,
        'all_probabilities': probs.tolist()
    }

    return result


def extract_embeddings(
    model_path,
    input_file,
    output_file,
    feature_order=None,
    batch_size=32,
    device='cuda'
):
    """
    Extract feature embeddings from trained model.

    Args:
        model_path (str): Path to trained model
        input_file (str): Path to input data
        output_file (str): Path to output file (.npy)
        feature_order (str): Path to feature order file
        batch_size (int): Batch size
        device (str): Device ('cuda' or 'cpu')
    """
    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = EpiAMLModel.load_model(model_path, device=device)
    model.eval()

    # Load data
    print(f"\nLoading data from {input_file}...")
    data, labels, _ = load_training_data(
        input_file,
        format='auto',
        binarize=True,
        feature_order=feature_order
    )

    print(f"  Loaded: {data.shape[0]} samples × {data.shape[1]} features")

    # Extract embeddings
    print(f"\nExtracting embeddings...")
    data_tensor = torch.FloatTensor(data).to(device)

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data_tensor[i:i + batch_size]
            embeddings = model.get_embeddings(batch)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings = np.concatenate(all_embeddings)

    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings
    np.save(output_file, embeddings)
    print(f"\nEmbeddings saved to {output_file}")

    # Also save labels if available
    if labels is not None:
        labels_file = output_file.replace('.npy', '_labels.npy')
        np.save(labels_file, labels)
        print(f"Labels saved to {labels_file}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using trained EpiAML model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:

  # Batch prediction
  python predict.py --model_path output/best_model.pt \\
                    --input_file test_data.h5 \\
                    --output_file predictions.csv \\
                    --feature_order ../cluster_output/feature_order.npy \\
                    --class_mapping output/class_mapping.csv

  # Single sample prediction
  python predict.py --model_path output/best_model.pt \\
                    --input_file single_sample.csv \\
                    --single \\
                    --class_mapping output/class_mapping.csv

  # Extract embeddings
  python predict.py --model_path output/best_model.pt \\
                    --input_file data.h5 \\
                    --extract_embeddings \\
                    --output_file embeddings.npy
        '''
    )

    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--input_file', required=True,
                        help='Path to input data (.h5 or .csv)')
    parser.add_argument('--output_file', default='predictions.csv',
                        help='Path to output file (default: predictions.csv)')

    parser.add_argument('--feature_order', default=None,
                        help='Path to feature order file (.npy, .json, .txt)')
    parser.add_argument('--class_mapping', default=None,
                        help='Path to class mapping CSV')

    parser.add_argument('--single', action='store_true',
                        help='Single sample prediction mode')
    parser.add_argument('--extract_embeddings', action='store_true',
                        help='Extract feature embeddings instead of predictions')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')

    args = parser.parse_args()

    if args.extract_embeddings:
        # Extract embeddings
        extract_embeddings(
            model_path=args.model_path,
            input_file=args.input_file,
            output_file=args.output_file,
            feature_order=args.feature_order,
            batch_size=args.batch_size,
            device=args.device
        )

    elif args.single:
        # Single sample prediction
        result = predict_single(
            model_path=args.model_path,
            input_file=args.input_file,
            class_mapping=args.class_mapping,
            feature_order=args.feature_order,
            device=args.device
        )

        print(f"\n{'='*70}")
        print(f"Prediction Result")
        print(f"{'='*70}")
        print(f"\nPredicted class: {result['predicted_class_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nTop 5 predictions:")
        for pred in result['top_predictions']:
            print(f"  {pred['class_name']:30s}: {pred['probability']:.4f}")

    else:
        # Batch prediction
        predict_batch(
            model_path=args.model_path,
            input_file=args.input_file,
            output_file=args.output_file,
            feature_order=args.feature_order,
            class_mapping=args.class_mapping,
            batch_size=args.batch_size,
            device=args.device
        )


if __name__ == '__main__':
    main()
