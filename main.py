"""
AI-Guided Instruction Set Extension for RISC-V
Streamlit Dashboard Application
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import sys
from pathlib import Path
import subprocess
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from profiler.torch_profiler import ModelProfiler
from isa_engine.isa_generator import ISAGenerator
from utils.analysis_tools import PerformanceAnalyzer
from rl_optimizer.isa_rl_agent import ISAOptimizer, run_rl_optimization
from compiler_integration.llvm_isa_backend import integrate_with_llvm
from workload_analyzer.multi_model_analyzer import WorkloadProfiler, run_multi_model_analysis
from advanced_analyzer.chat_interface import create_chat_interface
from advanced_analyzer.security_analyzer import analyze_instruction_security
from models.preloaded_models import model_manager, ModelCategory
from models.model_search import search_engine
from models.model_demo import model_demonstrator
from models.model_catalog import catalog_viewer

# Configure Streamlit page
st.set_page_config(
    page_title="RISC-V ISA Extension Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_data():
    """Load sample profiling and ISA data for demonstration"""
    # Sample profiling data
    profile_data = {
        'total_time': 0.245,
        'model_type': 'mobilenet_v2',
        'layers': [
            {'name': 'conv1', 'type': 'Conv2d', 'avg_time': 0.089, 'percentage': 36.3, 'parameters': 864, 'flops_estimate': 430000},
            {'name': 'conv2_1', 'type': 'Conv2d', 'avg_time': 0.054, 'percentage': 22.0, 'parameters': 18432, 'flops_estimate': 920000},
            {'name': 'relu1', 'type': 'ReLU', 'avg_time': 0.032, 'percentage': 13.1, 'parameters': 0, 'flops_estimate': 150000},
            {'name': 'conv2_2', 'type': 'Conv2d', 'avg_time': 0.028, 'percentage': 11.4, 'parameters': 9216, 'flops_estimate': 460000},
            {'name': 'batchnorm1', 'type': 'BatchNorm2d', 'avg_time': 0.021, 'percentage': 8.6, 'parameters': 128, 'flops_estimate': 32000},
            {'name': 'linear1', 'type': 'Linear', 'avg_time': 0.015, 'percentage': 6.1, 'parameters': 131072, 'flops_estimate': 131072},
            {'name': 'relu2', 'type': 'ReLU', 'avg_time': 0.006, 'percentage': 2.4, 'parameters': 0, 'flops_estimate': 1000}
        ],
        'bottlenecks': []
    }
    
    # Mark bottlenecks
    profile_data['bottlenecks'] = [layer for layer in profile_data['layers'] if layer['percentage'] > 10]
    
    # Sample ISA extensions
    isa_extensions = [
        {
            'name': 'VCONV.8',
            'description': 'Vectorized 2D Convolution - 8-bit integer convolution',
            'category': 'neural_compute',
            'operands': ['rs1', 'rs2', 'rd'],
            'opcode': '0x7B',
            'target_layer': 'conv1',
            'target_operation': 'Conv2d',
            'estimated_speedup': 4.2,
            'instruction_reduction': 71.5,
            'assembly_example': 'vconv.8 x10, x11, x13',
            'rationale': "Layer 'conv1' (Conv2d) consumes 36.3% of total execution time..."
        },
        {
            'name': 'VCONV.8',
            'description': 'Vectorized 2D Convolution - 8-bit integer convolution',
            'category': 'neural_compute',
            'operands': ['rs1', 'rs2', 'rd'],
            'opcode': '0x7B',
            'target_layer': 'conv2_1',
            'target_operation': 'Conv2d',
            'estimated_speedup': 3.8,
            'instruction_reduction': 65.0,
            'assembly_example': 'vconv.8 x10, x11, x13',
            'rationale': "Layer 'conv2_1' (Conv2d) consumes 22.0% of total execution time..."
        },
        {
            'name': 'RELU.V',
            'description': 'ReLU Activation Function - vectorized ReLU',
            'category': 'activation',
            'operands': ['rs1', 'rd'],
            'opcode': '0x7D',
            'target_layer': 'relu1',
            'target_operation': 'ReLU',
            'estimated_speedup': 2.4,
            'instruction_reduction': 48.0,
            'assembly_example': 'relu.v x10, x13',
            'rationale': "Layer 'relu1' (ReLU) consumes 13.1% of total execution time..."
        }
    ]
    
    return profile_data, isa_extensions

def create_performance_charts(profile_data, isa_extensions):
    """Create performance visualization charts"""
    
    # Layer execution time chart
    layer_df = pd.DataFrame(profile_data['layers'][:10])  # Top 10 layers
    
    fig_layers = px.bar(
        layer_df, 
        x='name', 
        y='percentage',
        title='Layer Execution Time Distribution',
        labels={'percentage': 'Execution Time (%)', 'name': 'Layer Name'},
        color='type',
        text='percentage'
    )
    fig_layers.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_layers.update_layout(xaxis_tickangle=-45)
    
    # ISA extension impact chart
    if isa_extensions:
        isa_df = pd.DataFrame(isa_extensions)
        
        fig_speedup = px.bar(
            isa_df,
            x='name',
            y='estimated_speedup',
            title='Estimated Speedup by ISA Extension',
            labels={'estimated_speedup': 'Speedup (x)', 'name': 'Instruction'},
            color='category',
            text='estimated_speedup'
        )
        fig_speedup.update_traces(texttemplate='%{text:.1f}x', textposition='outside')
        
        # Instruction reduction chart
        fig_reduction = px.bar(
            isa_df,
            x='name',
            y='instruction_reduction',
            title='Instruction Count Reduction',
            labels={'instruction_reduction': 'Reduction (%)', 'name': 'Instruction'},
            color='category',
            text='instruction_reduction'
        )
        fig_reduction.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        return fig_layers, fig_speedup, fig_reduction
    
    return fig_layers, None, None

# Main Streamlit application
def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔧 AI-Guided RISC-V ISA Extension Tool")
    st.markdown("Neural Network Profiling and Custom Instruction Generation")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Get models organized by category
    models_by_category = model_manager.get_models_by_category()
    
    # Model selection with categories
    model_selection_type = st.sidebar.selectbox(
        "Model Source",
        ["Preloaded Models (30+)", "Upload ONNX Model", "Sample Templates"]
    )
    
    # Model selection logic
    selected_model_id = None
    uploaded_file = None
    
    if model_selection_type == "Preloaded Models (30+)":
        st.sidebar.markdown("**🏗️ Choose from 30+ Preloaded Models**")
        
        # Category selection
        selected_category = st.sidebar.selectbox(
            "Model Category",
            list(models_by_category.keys()),
            help="Choose the type of neural network for your analysis"
        )
        
        # Model selection within category
        models_in_category = models_by_category[selected_category]
        model_display_names = []
        model_id_mapping = {}
        
        for model_id in models_in_category:
            info = model_manager.get_model_info(model_id)
            display_name = f"{info['name']} ({info['parameters']}, {info['complexity']})"
            model_display_names.append(display_name)
            model_id_mapping[display_name] = model_id
        
        selected_display_name = st.sidebar.selectbox(
            f"Select {selected_category} Model",
            model_display_names,
            help="Models show (parameters, complexity level)"
        )
        
        selected_model_id = model_id_mapping[selected_display_name]
        
        # Show model information
        model_info = model_manager.get_model_info(selected_model_id)
        
        with st.sidebar.expander("📋 Model Details", expanded=True):
            st.write(f"**Name:** {model_info['name']}")
            st.write(f"**Category:** {model_info['category']}")
            st.write(f"**Parameters:** {model_info['parameters']}")
            st.write(f"**Complexity:** {model_info['complexity']}")
            st.write(f"**Input Shape:** {model_info['input_shape']}")
            st.write(f"**Use Case:** {model_info['use_case']}")
            st.write(f"**Description:** {model_info['description']}")
        
        # Model search and recommendations
        with st.sidebar.expander("🔍 Model Search & Recommendations"):
            search_query = st.text_input("Search models:", placeholder="e.g., mobile, efficient, detection")
            
            if search_query:
                search_results = search_engine.search_models(search_query, selected_category)
                st.write(f"Found {len(search_results)} matching models:")
                for result_id in search_results[:5]:
                    result_info = model_manager.get_model_info(result_id)
                    st.write(f"• **{result_info['name']}** - {result_info['complexity']}")
            
            # Get recommendations based on use case
            st.write("**💡 Recommendations:**")
            recommendations = search_engine.get_recommendations(model_info['use_case'])
            for rec_id, reason in recommendations[:3]:
                if rec_id in model_manager.models_catalog:
                    rec_info = model_manager.get_model_info(rec_id)
                    st.write(f"• **{rec_info['name']}**: {reason}")
        
        # Performance constraints filter
        with st.sidebar.expander("⚡ Performance Constraints"):
            max_params = st.selectbox(
                "Max Parameters",
                [None, 1000000, 5000000, 25000000, 100000000],
                format_func=lambda x: "No limit" if x is None else f"{x/1000000:.0f}M params"
            )
            
            target_platform = st.selectbox(
                "Target Platform",
                [None, "mobile", "edge", "server"],
                format_func=lambda x: "Any platform" if x is None else x.title()
            )
            
            if st.button("🔧 Filter Models"):
                filtered_models = search_engine.filter_by_constraints(
                    max_params=max_params,
                    target_platform=target_platform
                )
                st.write(f"Found {len(filtered_models)} models matching constraints:")
                for filt_id in filtered_models[:5]:
                    filt_info = model_manager.get_model_info(filt_id)
                    st.write(f"• {filt_info['name']} ({filt_info['parameters']})")
    
    elif model_selection_type == "Upload ONNX Model":
        uploaded_file = st.sidebar.file_uploader(
            "Choose ONNX model file", 
            type=['onnx'],
            help="Upload an ONNX model file for analysis"
        )
    
    else:  # Sample Templates
        template_type = st.sidebar.selectbox(
            "Model Template",
            ["MobileNet-like", "ResNet-like", "Simple CNN"]
        )
    
    # Analysis options
    st.sidebar.header("🔍 Analysis Options")
    time_threshold = st.sidebar.slider(
        "Time Threshold for ISA Suggestions (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Generate ISA extensions for layers consuming more than this percentage of total time"
    )
    
    max_extensions = st.sidebar.slider(
        "Maximum ISA Extensions",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of ISA extensions to generate"
    )
    
    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
    
    # Main content
    if run_analysis or 'profile_data' not in st.session_state:
        with st.spinner("Running model profiling and ISA generation..."):
            try:
                if model_selection_type == "Preloaded Models (30+)" and selected_model_id:
                    # Load and profile preloaded model
                    model, model_metadata = model_manager.load_model(selected_model_id)
                    model_summary = model_manager.get_model_summary(selected_model_id)
                    
                    profiler = ModelProfiler()
                    input_shape = model_info['input_shape']
                    if len(input_shape) == 3:
                        sample_input = (1, *input_shape)
                    else:
                        sample_input = (1, *input_shape)
                    
                    profile_data = profiler.profile_pytorch_model(model, input_shape=sample_input)
                    profile_data['model_metadata'] = {
                        'id': selected_model_id,
                        'name': model_info['name'],
                        'category': model_info['category'],
                        'summary': model_summary
                    }
                    
                elif uploaded_file is not None:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Profile uploaded model
                    profiler = ModelProfiler()
                    profile_data = profiler.profile_model_from_file(temp_path)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                elif model_selection_type == "Sample Templates":
                    # Use sample templates
                    profiler = ModelProfiler()
                    if template_type == "ResNet-like":
                        model = profiler.create_sample_resnet()
                    elif template_type == "MobileNet-like":
                        model = profiler.create_sample_mobilenet()
                    else:
                        model = profiler.create_simple_cnn()
                    
                    profile_data = profiler.profile_pytorch_model(model, input_shape=(1, 3, 224, 224))
                    profile_data['model_metadata'] = {
                        'name': template_type,
                        'category': 'Sample Template'
                    }
                else:
                    # Use sample data for demo
                    profile_data, isa_extensions = load_sample_data()
                    
                    # Generate ISA extensions
                    generator = ISAGenerator()
                    isa_extensions = generator.generate_instructions(profile_data)['instructions']
                
                # Store in session state
                st.session_state.profile_data = profile_data
                st.session_state.isa_extensions = isa_extensions
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                # Fall back to sample data
                profile_data, isa_extensions = load_sample_data()
                st.session_state.profile_data = profile_data
                st.session_state.isa_extensions = isa_extensions
    
    # Use data from session state
    profile_data = st.session_state.get('profile_data')
    isa_extensions = st.session_state.get('isa_extensions')
    
    if not profile_data:
        profile_data, isa_extensions = load_sample_data()
    
    # Create tabs with advanced features including model catalog
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📚 Model Catalog",
        "📊 Profiling Results", 
        "🔧 ISA Extensions", 
        "💻 Emulator", 
        "📈 Performance Analysis",
        "🤖 RL Optimization",
        "🏗️ LLVM Integration",
        "🔍 Multi-Model Analysis",
        "🛡️ Security Analysis"
    ])
    
    with tab1:
        # Model Catalog Tab
        catalog_tab1, catalog_tab2, catalog_tab3, catalog_tab4 = st.tabs([
            "📖 Catalog Overview", 
            "🔍 Browse Models", 
            "⚖️ Compare Models", 
            "🤖 AI Recommendations"
        ])
        
        with catalog_tab1:
            catalog_viewer.show_catalog_overview()
        
        with catalog_tab2:
            catalog_viewer.show_interactive_browser()
        
        with catalog_tab3:
            catalog_viewer.show_comparison_tool()
        
        with catalog_tab4:
            catalog_viewer.show_recommendations_engine()
    
    with tab2:
        st.header("Model Profiling Results")
        
        # Enhanced model summary with preloaded model info
        if 'model_metadata' in profile_data:
            metadata = profile_data['model_metadata']
            
            # Model information header
            st.subheader(f"📊 {metadata.get('name', 'Unknown Model')}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Execution Time", 
                    f"{profile_data['total_time']:.3f}s"
                )
            
            with col2:
                st.metric(
                    "Model Category", 
                    metadata.get('category', 'Unknown')
                )
            
            with col3:
                layers_count = len(profile_data['layers'])
                st.metric("Layers Analyzed", layers_count)
            
            with col4:
                if 'summary' in metadata:
                    summary = metadata['summary']
                    params = summary.get('total_parameters', 0)
                    if params > 1000000:
                        param_str = f"{params/1000000:.1f}M"
                    elif params > 1000:
                        param_str = f"{params/1000:.1f}K"
                    else:
                        param_str = str(params)
                    st.metric("Parameters", param_str)
            
            # Detailed model information
            if 'summary' in metadata:
                summary = metadata['summary']
                
                # Create expandable sections for better organization
                with st.expander("🔍 Detailed Model Analysis", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Model Architecture:**")
                        st.write(f"• **Input Shape:** {summary.get('input_shape', 'Unknown')}")
                        st.write(f"• **Total Parameters:** {summary.get('total_parameters', 0):,}")
                        st.write(f"• **Trainable Parameters:** {summary.get('trainable_parameters', 0):,}")
                        st.write(f"• **Model Complexity:** {summary.get('complexity', 'Unknown')}")
                        
                        if 'estimated_flops' in summary:
                            flops = summary['estimated_flops']
                            if flops > 1e9:
                                flops_str = f"{flops/1e9:.2f} GFLOPs"
                            elif flops > 1e6:
                                flops_str = f"{flops/1e6:.2f} MFLOPs"
                            else:
                                flops_str = f"{flops/1e3:.2f} KFLOPs"
                            st.write(f"• **Estimated FLOPs:** {flops_str}")
                    
                    with col2:
                        st.write("**Performance Characteristics:**")
                        st.write(f"• **Use Case:** {summary.get('use_case', 'General purpose')}")
                        if 'memory_mb' in summary:
                            st.write(f"• **Memory Usage:** {summary['memory_mb']:.1f} MB")
                        
                        # ISA optimization potential
                        layers_count = len(profile_data['layers'])
                        if layers_count > 0:
                            conv_layers = len([l for l in profile_data['layers'] if 'conv' in l['type'].lower()])
                            linear_layers = len([l for l in profile_data['layers'] if 'linear' in l['type'].lower()])
                            
                            st.write("**ISA Optimization Potential:**")
                            if conv_layers > 0:
                                st.write(f"• **Convolution Layers:** {conv_layers} (High potential)")
                            if linear_layers > 0:
                                st.write(f"• **Linear Layers:** {linear_layers} (Medium potential)")
                            
                            total_compute_layers = conv_layers + linear_layers
                            if total_compute_layers > 0:
                                potential_score = min(100, (total_compute_layers / layers_count) * 100)
                                st.write(f"• **Optimization Score:** {potential_score:.1f}%")
                
                # Add interactive demonstrations
                with st.expander("🎮 Live Model Demo & Architecture Visualization"):
                    demo_tab1, demo_tab2, demo_tab3 = st.tabs(["Live Demo", "Architecture", "Performance Predictions"])
                    
                    with demo_tab1:
                        model_demonstrator.show_live_demo(metadata['id'])
                    
                    with demo_tab2:
                        model_demonstrator.show_model_architecture(metadata['id'])
                    
                    with demo_tab3:
                        model_demonstrator.show_performance_predictions(metadata['id'])
        
        else:
            # Basic model summary for non-preloaded models
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Execution Time", f"{profile_data['total_time']:.3f}s")
            with col2:
                st.metric("Number of Layers", len(profile_data['layers']))
            with col3:
                st.metric("Bottleneck Layers", len(profile_data['bottlenecks']))
            with col4:
                st.metric("Model Type", profile_data['model_type'].title())
        
        # Layer performance chart
        fig_layers, fig_speedup, fig_reduction = create_performance_charts(profile_data, isa_extensions)
        st.plotly_chart(fig_layers, use_container_width=True)
        
        # Detailed layer information
        st.subheader("Layer Details")
        layer_df = pd.DataFrame(profile_data['layers'])
        layer_df['avg_time'] = layer_df['avg_time'].round(4)
        layer_df['percentage'] = layer_df['percentage'].round(2)
        
        st.dataframe(
            layer_df,
            use_container_width=True,
            column_config={
                'avg_time': st.column_config.NumberColumn('Avg Time (s)', format="%.4f"),
                'percentage': st.column_config.NumberColumn('Time %', format="%.2f%%"),
                'parameters': st.column_config.NumberColumn('Parameters', format="%d"),
                'flops_estimate': st.column_config.NumberColumn('FLOPs Est.', format="%d")
            }
        )
    
    with tab3:
        st.header("Suggested ISA Extensions")
        
        if isa_extensions:
            # Performance impact charts
            col1, col2 = st.columns(2)
            with col1:
                if fig_speedup:
                    st.plotly_chart(fig_speedup, use_container_width=True)
            with col2:
                if fig_reduction:
                    st.plotly_chart(fig_reduction, use_container_width=True)
            
            # ISA extension details
            for i, ext in enumerate(isa_extensions[:max_extensions]):
                with st.expander(f"🔧 {ext['name']} - {ext['description']}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Target Operation:**", ext['target_operation'])
                        st.write("**Target Layer:**", ext['target_layer'])
                        st.write("**Category:**", ext['category'])
                        st.write("**Operands:**", ", ".join(ext['operands']))
                        st.write("**Rationale:**", ext['rationale'])
                    
                    with col2:
                        st.metric("Estimated Speedup", f"{ext['estimated_speedup']:.1f}x")
                        st.metric("Instruction Reduction", f"{ext['instruction_reduction']:.1f}%")
                        st.code(ext['assembly_example'], language='asm')
            
            # Export options
            st.subheader("Export Extensions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Assembly Code"):
                    generator = ISAGenerator()
                    assembly_code = generator.export_to_assembly(isa_extensions[:max_extensions])
                    st.code(assembly_code, language='asm')
            
            with col2:
                if st.button("📄 Download JSON"):
                    generator = ISAGenerator()
                    json_data = generator.export_to_json(isa_extensions[:max_extensions])
                    st.download_button(
                        label="Download ISA Extensions JSON",
                        data=json_data,
                        file_name="isa_extensions.json",
                        mime="application/json"
                    )
        else:
            st.info("No ISA extensions generated. Try running the analysis with different parameters.")
    
    with tab4:
        st.header("RISC-V Emulator")
        st.markdown("Simulate the performance of your custom ISA extensions")
        
        # Embed the emulator
        emulator_path = Path(__file__).parent / "emulator" / "web_riscv_emulator.html"
        
        if emulator_path.exists():
            # Read and display the emulator
            with open(emulator_path, 'r') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=1200, scrolling=True)
        else:
            st.error("Emulator not found. Please ensure web_riscv_emulator.html exists.")
            
            # Fallback: simple code editor
            st.subheader("Assembly Code Editor")
            sample_code = """# Sample RISC-V Assembly with Custom Extensions
li x10, 0x1000      # Input data address
li x11, 0x2000      # Weights address
vconv.8 x10, x11, x12    # Custom convolution instruction
relu.v x12, x13          # Custom ReLU instruction
"""
            
            assembly_code = st.text_area(
                "Enter RISC-V assembly code:",
                value=sample_code,
                height=300
            )
            
            if st.button("Simulate Execution"):
                st.info("Emulator simulation would run here...")
    
    with tab4:
        st.header("Performance Analysis")
        
        if isa_extensions:
            # Calculate overall performance improvement
            analyzer = PerformanceAnalyzer()
            analysis = analyzer.analyze_improvements(profile_data, isa_extensions)
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Overall Speedup", 
                    f"{analysis['overall_speedup']:.2f}x",
                    delta=f"{(analysis['overall_speedup'] - 1) * 100:.1f}%"
                )
            with col2:
                st.metric(
                    "Instruction Reduction", 
                    f"{analysis['total_instruction_reduction']:.1f}%"
                )
            with col3:
                st.metric(
                    "Energy Savings", 
                    f"{analysis['estimated_energy_savings']:.1f}%"
                )
            with col4:
                st.metric(
                    "Code Size Change", 
                    f"{analysis['code_size_change']:.1f}%"
                )
            
            # Detailed analysis
            st.subheader("Detailed Impact Analysis")
            
            impact_data = []
            for ext in isa_extensions[:max_extensions]:
                impact_data.append({
                    'Instruction': ext['name'],
                    'Target Layer': ext['target_layer'],
                    'Speedup': f"{ext['estimated_speedup']:.1f}x",
                    'Instruction Reduction': f"{ext['instruction_reduction']:.1f}%",
                    'Category': ext['category']
                })
            
            st.dataframe(pd.DataFrame(impact_data), use_container_width=True)
            
            # Performance timeline
            st.subheader("Execution Timeline Comparison")
            
            # Create timeline comparison chart
            timeline_data = {
                'Stage': ['Original Model', 'With ISA Extensions'],
                'Execution Time': [profile_data['total_time'], profile_data['total_time'] / analysis['overall_speedup']],
                'Instructions': [10000, 10000 * (1 - analysis['total_instruction_reduction']/100)]  # Estimated
            }
            
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Bar(
                name='Execution Time (s)',
                x=timeline_data['Stage'],
                y=timeline_data['Execution Time'],
                yaxis='y',
                offsetgroup=1
            ))
            fig_timeline.add_trace(go.Bar(
                name='Instruction Count',
                x=timeline_data['Stage'],
                y=timeline_data['Instructions'],
                yaxis='y2',
                offsetgroup=2
            ))
            
            fig_timeline.update_layout(
                title='Performance Comparison: Original vs ISA-Extended',
                xaxis=dict(title='Configuration'),
                yaxis=dict(title='Execution Time (s)', side='left'),
                yaxis2=dict(title='Instruction Count', side='right', overlaying='y'),
                barmode='group'
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        else:
            st.info("Performance analysis will be available after ISA extensions are generated.")
    
    with tab6:
        st.header("🤖 RL-Based ISA Optimization")
        st.markdown("Use reinforcement learning to automatically discover optimal instruction combinations")
        
        if st.button("🚀 Run RL Optimization", type="primary"):
            if profile_data:
                with st.spinner("Running RL agent to optimize ISA extensions..."):
                    try:
                        # Run RL optimization
                        rl_results = run_rl_optimization(profile_data, episodes=50)
                        
                        # Display results
                        st.success("RL optimization completed!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Reward", f"{rl_results['best_reward']:.2f}")
                        with col2:
                            st.metric("Instructions Found", len(rl_results['best_config']))
                        with col3:
                            st.metric("Training Episodes", len(rl_results['training_history']))
                        
                        # Show best configuration
                        st.subheader("Optimal ISA Configuration")
                        if rl_results['best_config']:
                            for i, instruction in enumerate(rl_results['best_config'], 1):
                                st.write(f"{i}. **{instruction}**")
                        else:
                            st.info("No optimal configuration found. Try adjusting parameters.")
                        
                        # Training progress
                        st.subheader("Training Progress")
                        training_data = rl_results['training_history']
                        if training_data:
                            df = pd.DataFrame(training_data)
                            
                            fig_training = px.line(df, x='episode', y='reward', 
                                                title='RL Training Progress')
                            st.plotly_chart(fig_training, use_container_width=True)
                            
                            # Show training statistics
                            st.write(f"Final reward: {training_data[-1]['reward']:.2f}")
                            st.write(f"Best episode: {max(training_data, key=lambda x: x['reward'])['episode']}")
                        
                        # Store results
                        st.session_state.rl_results = rl_results
                        
                    except Exception as e:
                        st.error(f"RL optimization failed: {str(e)}")
                        st.info("This feature requires neural network profiling data.")
            else:
                st.warning("Please run model profiling first to enable RL optimization.")
        
        # Show previous results if available
        if 'rl_results' in st.session_state:
            st.subheader("Previous RL Results")
            results = st.session_state.rl_results
            st.write(f"Best configuration: {', '.join(results['best_config'])}")
    
    with tab7:
        st.header("🏗️ LLVM Compiler Integration")
        st.markdown("Generate LLVM backend code for custom RISC-V instructions")
        
        if isa_extensions:
            if st.button("🔨 Generate LLVM Backend", type="primary"):
                with st.spinner("Generating LLVM integration files..."):
                    try:
                        # Generate LLVM backend
                        llvm_results = integrate_with_llvm(isa_extensions[:5])  # Limit to 5 instructions
                        
                        st.success("LLVM backend files generated successfully!")
                        
                        # Show generated files
                        st.subheader("Generated Files")
                        st.write(f"📁 **Backend Directory**: {llvm_results['backend_dir']}")
                        st.write(f"📄 **Source File**: {llvm_results['source_file']}")
                        st.write(f"🔧 **Compile Script**: {llvm_results['compile_script']}")
                        
                        # Show instructions
                        st.subheader("Instructions Integrated")
                        for instruction in llvm_results['instructions_used']:
                            st.write(f"• {instruction}")
                        
                        # Sample C++ code
                        st.subheader("Sample Implementation")
                        cpp_sample = f"""// Neural network with custom RISC-V instructions
#include <iostream>
#include <vector>

extern "C" {{
    void __builtin_riscv_vconv_8(uint8_t* input, uint8_t* weights, uint8_t* output, int h, int w, int c);
    void __builtin_riscv_relu_v(uint8_t* input, uint8_t* output, int size);
}}

int main() {{
    std::vector<uint8_t> input(224*224*3);
    std::vector<uint8_t> weights(64*3*7*7);
    std::vector<uint8_t> output(224*224*64);
    
    // Use custom convolution instruction
    __builtin_riscv_vconv_8(input.data(), weights.data(), output.data(), 224, 224, 3);
    
    // Use custom ReLU instruction
    __builtin_riscv_relu_v(output.data(), output.data(), output.size());
    
    std::cout << "Neural network executed with custom ISA!" << std::endl;
    return 0;
}}"""
                        st.code(cpp_sample, language='cpp')
                        
                        # Compilation instructions
                        st.subheader("Compilation Instructions")
                        st.code(f"""# Compile with RISC-V toolchain
riscv64-linux-gnu-g++ -march=rv64gc -mabi=lp64d -O2 \\
    -o neural_network {llvm_results['source_file']}

# Run with QEMU
qemu-riscv64 -L /usr/riscv64-linux-gnu neural_network""", language='bash')
                        
                    except Exception as e:
                        st.error(f"LLVM integration failed: {str(e)}")
        else:
            st.info("Generate ISA extensions first to enable LLVM integration.")
    
    with tab8:
        st.header("🔍 Multi-Model Workload Analysis")
        st.markdown("Compare optimization potential across different neural network types")
        
        if st.button("🔬 Run Multi-Model Analysis", type="primary"):
            with st.spinner("Analyzing CNN, Transformer, SNN, TinyML, and GAN workloads..."):
                try:
                    # Run multi-model analysis
                    workload_profiler = WorkloadProfiler()
                    analysis_results = workload_profiler.analyze_all_workloads()
                    
                    st.success("Multi-model analysis completed!")
                    
                    # Create comparison chart
                    comparison_data = []
                    for model_type, results in analysis_results.items():
                        if 'error' not in results:
                            opt_potential = results['optimization_potential']
                            comparison_data.append({
                                'Model Type': model_type,
                                'Max Speedup': opt_potential['max_theoretical_speedup'],
                                'ISA Potential': opt_potential['isa_extension_potential'],
                                'Bottleneck %': opt_potential['bottleneck_percentage'],
                                'Memory Optimization': opt_potential['memory_optimization_potential']
                            })
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        # Speedup potential chart
                        fig_speedup = px.bar(df_comparison, x='Model Type', y='Max Speedup',
                                           title='Maximum Theoretical Speedup by Model Type',
                                           color='Max Speedup')
                        st.plotly_chart(fig_speedup, use_container_width=True)
                        
                        # ISA potential chart
                        fig_isa = px.bar(df_comparison, x='Model Type', y='ISA Potential',
                                       title='ISA Extension Potential Score',
                                       color='ISA Potential')
                        st.plotly_chart(fig_isa, use_container_width=True)
                        
                        # Summary table
                        st.subheader("Detailed Comparison")
                        st.dataframe(df_comparison, use_container_width=True)
                        
                        # Generate recommendations
                        st.subheader("Optimization Recommendations")
                        best_model = max(comparison_data, key=lambda x: x['Max Speedup'])
                        st.success(f"**{best_model['Model Type']}** shows highest optimization potential "
                                 f"with {best_model['Max Speedup']:.1f}x theoretical speedup")
                        
                        for model_data in comparison_data:
                            model_type = model_data['Model Type']
                            if model_type == 'CNN':
                                st.info("🔧 CNNs benefit from convolution-specific instructions like VCONV.FUSED")
                            elif model_type == 'Transformer':
                                st.info("🔧 Transformers need matrix multiplication acceleration like MHATTN.BLOCK")
                            elif model_type == 'SNN':
                                st.info("🔧 SNNs require sparse computation instructions like SPIKE.THRESH")
                    
                    # Store results
                    st.session_state.multi_model_results = analysis_results
                    
                except Exception as e:
                    st.error(f"Multi-model analysis failed: {str(e)}")
                    st.info("Using sample data for demonstration")
        
        # Show previous results if available
        if 'multi_model_results' in st.session_state:
            st.subheader("Analysis Summary")
            results = st.session_state.multi_model_results
            st.write(f"Models analyzed: {len([r for r in results.values() if 'error' not in r])}")
    
    with tab9:
        st.header("🛡️ Security-Aware ISA Design")
        st.markdown("Analyze security risks and generate hardened instruction variants")
        
        # Environment selection
        environment = st.selectbox(
            "Target Deployment Environment",
            ["edge", "server", "mobile", "automotive"],
            help="Security requirements vary by deployment environment"
        )
        
        if isa_extensions and st.button("🔒 Analyze Security Risks", type="primary"):
            with st.spinner("Analyzing security risks and generating secure variants..."):
                try:
                    # Run security analysis
                    security_results = analyze_instruction_security(isa_extensions, environment)
                    
                    st.success("Security analysis completed!")
                    
                    # Summary metrics
                    summary = security_results['summary']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Instructions Analyzed", summary['total_analyzed'])
                    with col2:
                        st.metric("High Risk", summary['high_risk'])
                    with col3:
                        st.metric("Secure Variants", summary['secure_variants'])
                    with col4:
                        st.metric("Recommendations", summary['recommendations'])
                    
                    # Risk analysis
                    risks = security_results['risk_analysis']
                    
                    if risks['high_risk']:
                        st.subheader("⚠️ High-Risk Instructions")
                        for risk in risks['high_risk']:
                            with st.expander(f"🚨 {risk['instruction']}", expanded=True):
                                st.write("**Threats:**")
                                for threat in risk['risks']:
                                    st.write(f"• {threat.replace('_', ' ').title()}")
                                st.write("**Mitigation:**")
                                for mitigation in risk['mitigation']:
                                    st.write(f"• {mitigation}")
                    
                    # Secure variants
                    secure_instructions = security_results['secure_instructions']
                    if secure_instructions:
                        st.subheader("🔐 Security-Enhanced Variants")
                        for secure_inst in secure_instructions[:5]:
                            with st.expander(f"🛡️ {secure_inst['name']}"):
                                st.write(f"**Description:** {secure_inst['description']}")
                                st.write(f"**Security Features:** {', '.join(secure_inst.get('security_features', []))}")
                                st.write(f"**Performance Impact:** {secure_inst.get('estimated_speedup', 1.0):.1f}x")
                                if 'security_rationale' in secure_inst:
                                    st.write(f"**Rationale:** {secure_inst['security_rationale']}")
                                st.code(secure_inst.get('assembly_example', ''), language='asm')
                    
                    # Security recommendations
                    st.subheader("📋 Security Recommendations")
                    for i, rec in enumerate(risks['recommendations'], 1):
                        st.write(f"{i}. {rec}")
                    
                    # Export options
                    st.subheader("📤 Export Security Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Download Security Report"):
                            report = security_results['security_report']
                            st.download_button(
                                label="📄 Download Report",
                                data=report,
                                file_name="security_analysis_report.md",
                                mime="text/markdown"
                            )
                    
                    with col2:
                        if st.button("Download Secure Instructions"):
                            import json
                            secure_data = json.dumps(secure_instructions, indent=2)
                            st.download_button(
                                label="📋 Download JSON",
                                data=secure_data,
                                file_name="secure_isa_extensions.json",
                                mime="application/json"
                            )
                    
                    # Store results
                    st.session_state.security_results = security_results
                    
                except Exception as e:
                    st.error(f"Security analysis failed: {str(e)}")
        
        else:
            st.info("Generate ISA extensions first to enable security analysis.")
            
            # Show security best practices
            st.subheader("🔒 Security Best Practices")
            st.markdown("""
            **For Edge/IoT Deployments:**
            - Implement constant-time execution for all crypto operations
            - Add fault detection to critical instructions
            - Use control flow integrity for all branches
            - Implement secure boot and attestation
            
            **For Server Deployments:**
            - Focus on cache-timing attack prevention
            - Implement speculative execution barriers
            - Use memory protection and isolation
            - Add instruction-level integrity checking
            """)
    
    # Add chat interface at the bottom
    st.markdown("---")
    create_chat_interface()

if __name__ == "__main__":
    main()
