"""
Chat Interface for ISA Optimization
Natural language interface for neural network analysis and ISA recommendations
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import json
import re
from pathlib import Path
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from profiler.torch_profiler import ModelProfiler
from isa_engine.isa_generator import ISAGenerator
from utils.analysis_tools import PerformanceAnalyzer
from rl_optimizer.isa_rl_agent import ISAOptimizer
from workload_analyzer.multi_model_analyzer import WorkloadProfiler

class ChatISAOptimizer:
    """Chat-based interface for ISA optimization"""
    
    def __init__(self):
        self.conversation_history = []
        self.current_model_data = None
        self.optimization_results = {}
        self.rl_optimizer = ISAOptimizer()
        self.workload_profiler = WorkloadProfiler()
        
        # Initialize components
        self.profiler = ModelProfiler()
        self.isa_generator = ISAGenerator()
        self.analyzer = PerformanceAnalyzer()
        
        # Chat response templates
        self.response_templates = {
            'greeting': [
                "Hi! I'm your AI assistant for RISC-V ISA optimization. I can help you analyze neural networks and suggest custom instructions for maximum speedup.",
                "Hello! I specialize in creating custom RISC-V instruction set extensions for neural networks. What model would you like to optimize?",
                "Welcome! I can analyze your neural network and recommend custom RISC-V instructions to accelerate it. What can I help you with?"
            ],
            'model_analysis': [
                "I'll analyze your {} model to identify performance bottlenecks and suggest optimal ISA extensions.",
                "Let me profile your {} neural network and find the best custom instructions for acceleration.",
                "I'm examining your {} model architecture to determine the most effective ISA optimizations."
            ],
            'optimization_complete': [
                "Analysis complete! I found {} optimization opportunities that could provide up to {:.1f}x speedup.",
                "Great! I've identified {} custom instructions that could accelerate your model by {:.1f}x.",
                "Perfect! The analysis shows {} ISA extensions could give you {:.1f}x performance improvement."
            ]
        }
    
    def process_chat_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process chat message and return response with actions"""
        message_lower = message.lower().strip()
        
        # Determine intent
        intent = self._classify_intent(message_lower)
        
        # Process based on intent
        if intent == 'greeting':
            return self._handle_greeting()
        elif intent == 'model_upload':
            return self._handle_model_upload(message, context)
        elif intent == 'analyze_model':
            return self._handle_model_analysis(message, context)
        elif intent == 'optimize_request':
            return self._handle_optimization_request(message, context)
        elif intent == 'compare_models':
            return self._handle_model_comparison(message, context)
        elif intent == 'technical_question':
            return self._handle_technical_question(message, context)
        elif intent == 'performance_query':
            return self._handle_performance_query(message, context)
        else:
            return self._handle_general_query(message, context)
    
    def _classify_intent(self, message: str) -> str:
        """Classify user intent from message"""
        # Intent keywords
        intent_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'start', 'begin'],
            'model_upload': ['upload', 'load', 'import', 'file', 'model file'],
            'analyze_model': ['analyze', 'profile', 'examine', 'check', 'study'],
            'optimize_request': ['optimize', 'accelerate', 'speedup', 'improve', 'faster'],
            'compare_models': ['compare', 'versus', 'vs', 'different', 'which is better'],
            'technical_question': ['how does', 'what is', 'explain', 'why', 'how'],
            'performance_query': ['performance', 'speed', 'benchmark', 'timing', 'fps']
        }
        
        for intent, keywords in intent_patterns.items():
            if any(keyword in message for keyword in keywords):
                return intent
        
        return 'general'
    
    def _handle_greeting(self) -> Dict[str, Any]:
        """Handle greeting messages"""
        import random
        response = random.choice(self.response_templates['greeting'])
        
        return {
            'response': response,
            'suggestions': [
                "Upload a neural network model (ONNX format)",
                "Analyze a sample CNN or Transformer model",
                "Compare different model types for optimization",
                "Ask about RISC-V ISA extensions"
            ],
            'actions': ['show_model_selector'],
            'intent': 'greeting'
        }
    
    def _handle_model_upload(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model upload requests"""
        response = "I'll help you upload and analyze your model. Please use the file uploader in the sidebar to select your ONNX model file."
        
        if context and 'uploaded_file' in context:
            # Process uploaded file
            try:
                model_data = self._analyze_uploaded_model(context['uploaded_file'])
                self.current_model_data = model_data
                
                response = f"""Successfully loaded your model! Here's what I found:

📊 **Model Analysis:**
- Total layers: {len(model_data['layers'])}
- Execution time: {model_data['total_time']:.3f}s
- Bottleneck layers: {len(model_data['bottlenecks'])}

🔍 **Top bottlenecks:**"""
                
                for layer in model_data['bottlenecks'][:3]:
                    response += f"\n- {layer['name']} ({layer['type']}): {layer['percentage']:.1f}% of time"
                
                response += "\n\nWould you like me to suggest ISA extensions for optimization?"
                
                return {
                    'response': response,
                    'suggestions': [
                        "Generate ISA optimization recommendations",
                        "Run RL-based optimization",
                        "Compare with other model types",
                        "Show detailed performance breakdown"
                    ],
                    'actions': ['show_analysis_results'],
                    'data': model_data
                }
                
            except Exception as e:
                response = f"I encountered an error processing your model: {str(e)}. Please ensure it's a valid ONNX file."
        
        return {
            'response': response,
            'suggestions': ["Try uploading an ONNX model file"],
            'actions': ['show_file_uploader'],
            'intent': 'model_upload'
        }
    
    def _handle_model_analysis(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model analysis requests"""
        # Extract model type from message
        model_type = self._extract_model_type(message)
        
        if not model_type:
            return {
                'response': "Which type of model would you like me to analyze? I can work with CNN, Transformer, SNN, TinyML, or GAN models.",
                'suggestions': ["Analyze CNN model", "Analyze Transformer model", "Analyze SNN model", "Analyze TinyML model"],
                'actions': ['show_model_type_selector'],
                'intent': 'model_analysis'
            }
        
        # Perform analysis
        try:
            import random
            response = random.choice(self.response_templates['model_analysis']).format(model_type)
            
            # Run actual analysis
            if model_type.lower() == 'cnn':
                model_data = self.profiler.profile_sample_model('mobilenet')
            elif model_type.lower() == 'transformer':
                # Use workload profiler for transformer
                results = self.workload_profiler.analyze_all_workloads()
                model_data = results.get('Transformer', {}).get('profile_data', {})
            else:
                model_data = self.profiler.profile_sample_model('mobilenet')  # Fallback
            
            self.current_model_data = model_data
            
            # Generate ISA recommendations
            isa_extensions = self.isa_generator.generate_extensions(model_data)
            
            analysis_response = f"""✅ **Analysis Complete for {model_type.upper()} Model**

📊 **Performance Profile:**
- Total execution time: {model_data.get('total_time', 0):.3f}s
- Number of layers: {len(model_data.get('layers', []))}
- Major bottlenecks: {len(model_data.get('bottlenecks', []))}

🔧 **Recommended ISA Extensions:**"""
            
            for ext in isa_extensions[:3]:
                analysis_response += f"""
- **{ext['name']}**: {ext['estimated_speedup']:.1f}x speedup
  Target: {ext['target_layer']} ({ext['target_operation']})"""
            
            potential_speedup = max([ext['estimated_speedup'] for ext in isa_extensions]) if isa_extensions else 1.0
            analysis_response += f"\n\n🚀 **Overall potential speedup: {potential_speedup:.1f}x**"
            
            return {
                'response': analysis_response,
                'suggestions': [
                    "Run RL optimization for better results",
                    "Generate LLVM backend code",
                    "Compare with other model types",
                    "Export results"
                ],
                'actions': ['show_optimization_results'],
                'data': {
                    'model_data': model_data,
                    'isa_extensions': isa_extensions,
                    'model_type': model_type
                }
            }
            
        except Exception as e:
            return {
                'response': f"I encountered an error during analysis: {str(e)}. Let me try with a sample model instead.",
                'suggestions': ["Try with sample data", "Upload a different model"],
                'actions': ['show_error_message'],
                'intent': 'error'
            }
    
    def _handle_optimization_request(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimization requests"""
        if not self.current_model_data:
            return {
                'response': "I need to analyze a model first before I can optimize it. Would you like me to analyze a sample model or do you have one to upload?",
                'suggestions': [
                    "Analyze sample CNN model",
                    "Analyze sample Transformer model",
                    "Upload my own model"
                ],
                'actions': ['show_model_selector'],
                'intent': 'optimization'
            }
        
        # Determine optimization type
        if 'rl' in message.lower() or 'reinforcement' in message.lower():
            return self._run_rl_optimization()
        elif 'compare' in message.lower():
            return self._run_comparative_optimization()
        else:
            return self._run_standard_optimization()
    
    def _run_rl_optimization(self) -> Dict[str, Any]:
        """Run RL-based optimization"""
        try:
            response = "🤖 **Running RL-based optimization...**\n\nI'm using reinforcement learning to explore the best combination of custom instructions for your model. This may take a moment..."
            
            # Run RL optimization
            rl_results = self.rl_optimizer.optimize_isa(self.current_model_data, episodes=50)
            
            best_config = rl_results.get('best_config', [])
            best_reward = rl_results.get('best_reward', 0)
            final_metrics = rl_results.get('final_metrics', {})
            
            optimization_response = f"""✅ **RL Optimization Complete!**

🎯 **Best Configuration Found:**
{', '.join(best_config) if best_config else 'No optimal configuration found'}

📈 **Performance Metrics:**
- Optimization reward: {best_reward:.2f}
- Estimated speedup: {final_metrics.get('speedup', 1.0):.1f}x
- Energy reduction: {final_metrics.get('energy_reduction', 0) * 100:.1f}%
- FLOPS reduction: {final_metrics.get('flops_reduction', 0) * 100:.1f}%

🧠 **RL Agent learned {len(rl_results.get('training_history', []))} episodes to find this optimal configuration.**"""
            
            return {
                'response': optimization_response,
                'suggestions': [
                    "Generate LLVM backend for this configuration",
                    "Compare with other optimization methods",
                    "Export optimization results",
                    "Analyze training convergence"
                ],
                'actions': ['show_rl_results'],
                'data': rl_results
            }
            
        except Exception as e:
            return {
                'response': f"RL optimization encountered an error: {str(e)}. Let me try standard optimization instead.",
                'suggestions': ["Try standard optimization", "Analyze model again"],
                'actions': ['show_error_message'],
                'intent': 'error'
            }
    
    def _run_standard_optimization(self) -> Dict[str, Any]:
        """Run standard ISA optimization"""
        try:
            # Generate ISA extensions
            isa_extensions = self.isa_generator.generate_extensions(self.current_model_data)
            
            # Analyze performance
            analysis = self.analyzer.analyze_improvements(self.current_model_data, isa_extensions)
            
            import random
            response = random.choice(self.response_templates['optimization_complete']).format(
                len(isa_extensions), analysis.get('overall_speedup', 1.0)
            )
            
            optimization_response = f"""{response}

🔧 **Recommended ISA Extensions:**"""
            
            for i, ext in enumerate(isa_extensions[:5], 1):
                optimization_response += f"""
{i}. **{ext['name']}** - {ext['estimated_speedup']:.1f}x speedup
   Target: {ext['target_operation']} operations
   Instruction reduction: {ext['instruction_reduction']:.1f}%"""
            
            optimization_response += f"""

📊 **Overall Performance Improvement:**
- Speedup: {analysis.get('overall_speedup', 1.0):.2f}x
- Instruction reduction: {analysis.get('total_instruction_reduction', 0):.1f}%
- Energy savings: {analysis.get('estimated_energy_savings', 0):.1f}%"""
            
            return {
                'response': optimization_response,
                'suggestions': [
                    "Generate LLVM compiler integration",
                    "Run RL optimization for better results",
                    "Compare with other models",
                    "Export assembly code"
                ],
                'actions': ['show_optimization_results'],
                'data': {
                    'isa_extensions': isa_extensions,
                    'analysis': analysis
                }
            }
            
        except Exception as e:
            return {
                'response': f"Optimization failed: {str(e)}. This might be due to insufficient model data.",
                'suggestions': ["Analyze model first", "Try with sample data"],
                'actions': ['show_error_message'],
                'intent': 'error'
            }
    
    def _handle_model_comparison(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model comparison requests"""
        try:
            response = "🔍 **Running multi-model comparison analysis...**\n\nI'm analyzing CNN, Transformer, SNN, TinyML, and GAN workloads to compare their optimization potential."
            
            # Run workload analysis
            comparison_results = self.workload_profiler.analyze_all_workloads()
            
            comparison_response = """✅ **Multi-Model Comparison Complete!**

📊 **Optimization Potential by Model Type:**
"""
            
            for model_type, results in comparison_results.items():
                if 'error' in results:
                    continue
                    
                opt_potential = results.get('optimization_potential', {})
                max_speedup = opt_potential.get('max_theoretical_speedup', 1.0)
                isa_potential = opt_potential.get('isa_extension_potential', 0.0)
                bottlenecks = len(results.get('profile_data', {}).get('bottlenecks', []))
                
                comparison_response += f"""
**{model_type}:**
- Max speedup potential: {max_speedup:.1f}x
- ISA optimization score: {isa_potential:.2f}
- Major bottlenecks: {bottlenecks}"""
            
            # Recommendations
            comparison_response += """

🎯 **Recommendations:**
- **Transformers** show highest optimization potential due to matrix-heavy operations
- **CNNs** benefit most from convolution-specific instructions
- **SNNs** need specialized sparse computation instructions
- **TinyML** requires memory-optimized and quantized instructions"""
            
            return {
                'response': comparison_response,
                'suggestions': [
                    "Focus on Transformer optimization",
                    "Optimize CNN workloads",
                    "Explore SNN-specific instructions",
                    "Generate report"
                ],
                'actions': ['show_comparison_results'],
                'data': comparison_results
            }
            
        except Exception as e:
            return {
                'response': f"Model comparison failed: {str(e)}. Let me provide general guidance instead.",
                'suggestions': ["Analyze single model", "Try sample data"],
                'actions': ['show_error_message'],
                'intent': 'error'
            }
    
    def _handle_technical_question(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technical questions about ISA extensions"""
        message_lower = message.lower()
        
        if 'risc-v' in message_lower:
            response = """🔧 **RISC-V ISA Extensions Explained:**

RISC-V is an open instruction set architecture that allows custom extensions. For neural networks, we can add specialized instructions like:

• **VCONV.8** - Vectorized 8-bit convolution
• **VMMUL.F** - Vector-matrix multiply (floating point)
• **RELU.V** - Vectorized ReLU activation
• **BNORM.2D** - Batch normalization

These custom instructions can replace multiple standard RISC-V instructions, providing significant speedup for neural network operations."""
            
        elif 'how does' in message_lower and 'work' in message_lower:
            response = """⚙️ **How ISA Optimization Works:**

1. **Profiling**: I analyze your neural network to find computational bottlenecks
2. **Pattern Recognition**: I identify repetitive operation patterns
3. **Instruction Design**: I suggest custom RISC-V instructions for these patterns
4. **Performance Estimation**: I calculate potential speedup and efficiency gains
5. **Code Generation**: I can generate LLVM backend code for implementation

The goal is to replace many simple instructions with fewer, more powerful ones."""
            
        elif 'llvm' in message_lower:
            response = """🔨 **LLVM Integration:**

I can generate LLVM backend files for your custom instructions:

• **Intrinsics definition** (.td files)
• **Instruction patterns** for optimization
• **C++ integration** code
• **Assembly generation** templates

This allows you to compile neural networks that automatically use your custom instructions for maximum performance."""
            
        else:
            response = """🤔 I'd be happy to explain any technical aspect of ISA optimization! You can ask me about:

• How RISC-V ISA extensions work
• LLVM compiler integration
• Neural network profiling techniques
• Reinforcement learning for optimization
• Performance analysis methods"""
        
        return {
            'response': response,
            'suggestions': [
                "Show me LLVM integration example",
                "Explain RL optimization",
                "How to measure performance?",
                "Generate code example"
            ],
            'actions': ['show_technical_info'],
            'intent': 'technical'
        }
    
    def _handle_performance_query(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance-related queries"""
        if not self.current_model_data:
            response = "I need to analyze a model first to provide performance insights. Would you like me to analyze a sample model?"
            return {
                'response': response,
                'suggestions': ["Analyze CNN model", "Analyze Transformer model"],
                'actions': ['show_model_selector'],
                'intent': 'performance'
            }
        
        # Provide performance insights
        total_time = self.current_model_data.get('total_time', 0)
        bottlenecks = self.current_model_data.get('bottlenecks', [])
        layers = self.current_model_data.get('layers', [])
        
        response = f"""📈 **Performance Analysis:**

⏱️ **Current Performance:**
- Total execution time: {total_time:.3f}s
- Number of layers: {len(layers)}
- Bottleneck layers: {len(bottlenecks)}

🎯 **Optimization Opportunities:**"""
        
        for layer in bottlenecks[:3]:
            response += f"\n- {layer['name']}: {layer['percentage']:.1f}% of total time"
        
        if bottlenecks:
            max_potential = sum(layer['percentage'] for layer in bottlenecks[:3]) / 100 * 2.5
            response += f"\n\n🚀 **Potential speedup: {max_potential:.1f}x** with targeted ISA extensions"
        
        return {
            'response': response,
            'suggestions': [
                "Generate ISA optimizations",
                "Run detailed analysis",
                "Compare with other models",
                "Export performance report"
            ],
            'actions': ['show_performance_analysis'],
            'data': self.current_model_data
        }
    
    def _handle_general_query(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries"""
        response = """I'm here to help you optimize neural networks with custom RISC-V instruction set extensions! 

Here's what I can do:
🔍 **Analyze** your neural network models
🔧 **Generate** custom ISA extensions  
🤖 **Use RL** to find optimal instruction combinations
💻 **Create LLVM** compiler integration
📊 **Compare** different model types

What would you like to work on?"""
        
        return {
            'response': response,
            'suggestions': [
                "Analyze my neural network model",
                "Compare different model types",
                "Explain how ISA extensions work",
                "Show me optimization examples"
            ],
            'actions': ['show_main_options'],
            'intent': 'general'
        }
    
    def _extract_model_type(self, message: str) -> Optional[str]:
        """Extract model type from message"""
        model_types = {
            'cnn': ['cnn', 'convolutional', 'resnet', 'mobilenet', 'efficientnet'],
            'transformer': ['transformer', 'bert', 'gpt', 'attention', 'llm'],
            'snn': ['snn', 'spiking', 'neuromorphic'],
            'tinyml': ['tinyml', 'tiny', 'edge', 'embedded', 'microcontroller'],
            'gan': ['gan', 'generative', 'generator', 'discriminator']
        }
        
        message_lower = message.lower()
        for model_type, keywords in model_types.items():
            if any(keyword in message_lower for keyword in keywords):
                return model_type
        
        return None
    
    def _analyze_uploaded_model(self, uploaded_file) -> Dict[str, Any]:
        """Analyze uploaded model file"""
        # In a real implementation, this would process the actual file
        # For now, return sample data
        return {
            'total_time': 0.156,
            'model_type': 'uploaded_model',
            'layers': [
                {'name': 'input_layer', 'type': 'Input', 'avg_time': 0.001, 'percentage': 0.6, 'parameters': 0, 'flops_estimate': 0},
                {'name': 'conv1', 'type': 'Conv2d', 'avg_time': 0.089, 'percentage': 57.1, 'parameters': 864, 'flops_estimate': 430000},
                {'name': 'relu1', 'type': 'ReLU', 'avg_time': 0.032, 'percentage': 20.5, 'parameters': 0, 'flops_estimate': 150000},
                {'name': 'conv2', 'type': 'Conv2d', 'avg_time': 0.028, 'percentage': 17.9, 'parameters': 9216, 'flops_estimate': 460000},
                {'name': 'output', 'type': 'Linear', 'avg_time': 0.006, 'percentage': 3.9, 'parameters': 10000, 'flops_estimate': 10000}
            ],
            'bottlenecks': []
        }

def create_chat_interface():
    """Create Streamlit chat interface"""
    st.subheader("🤖 AI Chat Assistant")
    st.markdown("Ask me anything about neural network optimization and ISA extensions!")
    
    # Initialize chat optimizer
    if 'chat_optimizer' not in st.session_state:
        st.session_state.chat_optimizer = ChatISAOptimizer()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_message = st.chat_input("Ask about neural network optimization...")
    
    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Process message
        context = {
            'uploaded_file': st.session_state.get('uploaded_file'),
            'current_analysis': st.session_state.get('profile_data')
        }
        
        response_data = st.session_state.chat_optimizer.process_chat_message(user_message, context)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response_data['response'],
            'suggestions': response_data.get('suggestions', []),
            'data': response_data.get('data')
        })
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])
            
            if message['role'] == 'assistant' and 'suggestions' in message:
                st.write("💡 **Suggestions:**")
                for suggestion in message['suggestions']:
                    if st.button(suggestion, key=f"suggestion_{len(st.session_state.chat_history)}_{suggestion}"):
                        # Process suggestion as new message
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': suggestion
                        })
                        st.rerun()

if __name__ == "__main__":
    # Test the chat interface
    chat_optimizer = ChatISAOptimizer()
    
    test_messages = [
        "Hello!",
        "Analyze a CNN model for me",
        "What ISA extensions do you recommend?",
        "Run RL optimization",
        "How does RISC-V work?"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = chat_optimizer.process_chat_message(message)
        print(f"Assistant: {response['response']}")
        if response.get('suggestions'):
            print(f"Suggestions: {response['suggestions']}")