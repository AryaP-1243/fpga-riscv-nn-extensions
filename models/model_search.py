"""
Model Search and Filtering Functionality
Provides search, filter, and recommendation capabilities for preloaded models
"""

from typing import Dict, List, Tuple, Any
from models.preloaded_models import model_manager, ModelCategory
import re

class ModelSearchEngine:
    """Search and filter engine for neural network models"""
    
    def __init__(self):
        self.models = model_manager.models_catalog
        self.categories = model_manager.get_models_by_category()
    
    def search_models(self, query: str, category: str = None, complexity: str = None) -> List[str]:
        """Search models based on query, category, and complexity"""
        query = query.lower().strip()
        results = []
        
        for model_id, info in self.models.items():
            # Category filter
            if category and info['category'] != category:
                continue
            
            # Complexity filter  
            if complexity and info['complexity'] != complexity:
                continue
            
            # Text search in name, description, use_case
            search_text = f"{info['name']} {info['description']} {info['use_case']}".lower()
            
            if not query or query in search_text:
                results.append(model_id)
        
        return results
    
    def get_recommendations(self, use_case: str) -> List[Tuple[str, str]]:
        """Get model recommendations based on use case"""
        recommendations = []
        use_case = use_case.lower()
        
        if "mobile" in use_case or "edge" in use_case or "embedded" in use_case:
            # Mobile/Edge recommendations
            mobile_models = [
                ("mobilenet_v3_small", "Ultra-efficient for extreme edge deployment"),
                ("mobilenet_v2", "Balanced efficiency for mobile apps"),
                ("squeezenet1_0", "Minimal parameters for constrained devices"),
                ("efficientnet_b0", "Excellent accuracy-efficiency tradeoff")
            ]
            recommendations.extend(mobile_models)
        
        elif "accuracy" in use_case or "performance" in use_case or "server" in use_case:
            # High accuracy recommendations
            accuracy_models = [
                ("resnet101", "Maximum accuracy for server deployment"),
                ("resnet50", "Balanced accuracy and speed"),
                ("efficientnet_b2", "Scaled efficiency with higher accuracy"),
                ("inception_v3", "Multi-scale feature analysis")
            ]
            recommendations.extend(accuracy_models)
        
        elif "detection" in use_case or "object" in use_case:
            # Object detection recommendations
            detection_models = [
                ("yolov5_nano", "Real-time lightweight detection"),
                ("ssd_mobilenet", "Mobile object detection"),
                ("faster_rcnn_backbone", "High accuracy two-stage detection")
            ]
            recommendations.extend(detection_models)
        
        elif "segmentation" in use_case or "semantic" in use_case:
            # Segmentation recommendations
            segmentation_models = [
                ("unet", "Medical and precise segmentation"),
                ("deeplabv3_resnet50", "High-quality semantic segmentation"),
                ("fcn_resnet50", "General purpose segmentation")
            ]
            recommendations.extend(segmentation_models)
        
        elif "text" in use_case or "ocr" in use_case:
            # OCR/Text recommendations
            ocr_models = [
                ("crnn_text", "Scene text recognition"),
                ("transformer_ocr", "Document digitization")
            ]
            recommendations.extend(ocr_models)
        
        elif "generation" in use_case or "creative" in use_case:
            # Generation recommendations
            gen_models = [
                ("stylegan2_generator", "High-quality face generation"),
                ("cyclegan_generator", "Style transfer and domain adaptation"),
                ("diffusion_unet", "Text-to-image generation")
            ]
            recommendations.extend(gen_models)
        
        else:
            # General recommendations
            general_models = [
                ("resnet18", "Reliable baseline for classification"),
                ("mobilenet_v2", "Efficient general purpose model"),
                ("efficientnet_b0", "Modern efficient architecture")
            ]
            recommendations.extend(general_models)
        
        # Filter out models not in catalog and limit results
        valid_recommendations = [
            (model_id, reason) for model_id, reason in recommendations 
            if model_id in self.models
        ]
        
        return valid_recommendations[:6]  # Top 6 recommendations
    
    def get_similar_models(self, model_id: str) -> List[Tuple[str, str, float]]:
        """Find similar models based on architecture characteristics"""
        if model_id not in self.models:
            return []
        
        reference_model = self.models[model_id]
        similar_models = []
        
        for other_id, other_info in self.models.items():
            if other_id == model_id:
                continue
            
            similarity_score = self._calculate_similarity(reference_model, other_info)
            
            if similarity_score > 0.3:  # Threshold for similarity
                reason = self._get_similarity_reason(reference_model, other_info)
                similar_models.append((other_id, reason, similarity_score))
        
        # Sort by similarity score
        similar_models.sort(key=lambda x: x[2], reverse=True)
        return similar_models[:5]
    
    def _calculate_similarity(self, model1: Dict, model2: Dict) -> float:
        """Calculate similarity score between two models"""
        score = 0.0
        
        # Category similarity (high weight)
        if model1['category'] == model2['category']:
            score += 0.4
        
        # Complexity similarity
        complexity_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5, 'Extreme': 6}
        c1 = complexity_map.get(model1['complexity'], 3)
        c2 = complexity_map.get(model2['complexity'], 3)
        complexity_diff = abs(c1 - c2)
        score += max(0, (6 - complexity_diff) / 6) * 0.3
        
        # Parameter count similarity (parse from string)
        def parse_params(param_str):
            if 'M' in param_str:
                return float(param_str.replace('M', '')) * 1e6
            elif 'K' in param_str:
                return float(param_str.replace('K', '')) * 1e3
            else:
                try:
                    return float(param_str)
                except:
                    return 10e6  # Default
        
        p1 = parse_params(model1['parameters'])
        p2 = parse_params(model2['parameters'])
        
        if p1 > 0 and p2 > 0:
            param_ratio = min(p1, p2) / max(p1, p2)
            score += param_ratio * 0.2
        
        # Use case similarity
        use_case_words1 = set(model1['use_case'].lower().split())
        use_case_words2 = set(model2['use_case'].lower().split())
        common_words = use_case_words1.intersection(use_case_words2)
        if common_words:
            score += len(common_words) / max(len(use_case_words1), len(use_case_words2)) * 0.1
        
        return score
    
    def _get_similarity_reason(self, model1: Dict, model2: Dict) -> str:
        """Generate human-readable reason for similarity"""
        reasons = []
        
        if model1['category'] == model2['category']:
            reasons.append(f"Same category ({model1['category']})")
        
        if model1['complexity'] == model2['complexity']:
            reasons.append(f"Similar complexity ({model1['complexity']})")
        
        # Check for architecture family
        model1_name = model1['name'].lower()
        model2_name = model2['name'].lower()
        
        if any(arch in model1_name and arch in model2_name for arch in ['resnet', 'mobilenet', 'efficientnet', 'vgg']):
            reasons.append("Same architecture family")
        
        if not reasons:
            reasons.append("Similar characteristics")
        
        return "; ".join(reasons)
    
    def filter_by_constraints(self, max_params: int = None, max_flops: int = None, 
                            min_accuracy: float = None, target_platform: str = None) -> List[str]:
        """Filter models by computational constraints"""
        filtered_models = []
        
        for model_id in self.models:
            try:
                summary = model_manager.get_model_summary(model_id)
                
                # Parameter constraint
                if max_params and summary['total_parameters'] > max_params:
                    continue
                
                # FLOP constraint
                if max_flops and summary['estimated_flops'] > max_flops:
                    continue
                
                # Platform-specific filtering
                if target_platform:
                    info = self.models[model_id]
                    if target_platform.lower() == 'mobile' and info['complexity'] in ['High', 'Very High', 'Extreme']:
                        continue
                    elif target_platform.lower() == 'edge' and info['complexity'] in ['Very High', 'Extreme']:
                        continue
                
                filtered_models.append(model_id)
                
            except Exception:
                # Skip models that can't be analyzed
                continue
        
        return filtered_models
    
    def get_performance_comparison(self, model_ids: List[str]) -> List[Dict]:
        """Compare performance characteristics of multiple models"""
        comparison = []
        
        for model_id in model_ids:
            if model_id not in self.models:
                continue
            
            try:
                info = self.models[model_id]
                summary = model_manager.get_model_summary(model_id)
                
                comparison.append({
                    'model_id': model_id,
                    'name': info['name'],
                    'category': info['category'],
                    'parameters': summary['total_parameters'],
                    'flops': summary['estimated_flops'],
                    'memory_mb': summary['memory_mb'],
                    'complexity': info['complexity'],
                    'use_case': info['use_case']
                })
            except Exception:
                continue
        
        return comparison

# Global search engine instance
search_engine = ModelSearchEngine()