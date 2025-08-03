"""
Interactive Model Catalog Viewer
Provides comprehensive browsing and comparison interface for all preloaded models
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
from models.preloaded_models import model_manager, ModelCategory
from models.model_search import search_engine

class ModelCatalogViewer:
    """Interactive catalog for browsing and comparing neural network models"""
    
    def __init__(self):
        self.models = model_manager.models_catalog
        self.categories = model_manager.get_models_by_category()
    
    def show_catalog_overview(self) -> None:
        """Display comprehensive catalog overview"""
        st.header("📚 Neural Network Model Catalog")
        st.markdown("**30+ Preloaded Models Across All Major Categories**")
        
        # Statistics overview
        total_models = len(self.models)
        categories = len(self.categories)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", total_models)
        with col2:
            st.metric("Categories", categories)
        with col3:
            complexity_counts = {}
            for info in self.models.values():
                complexity_counts[info['complexity']] = complexity_counts.get(info['complexity'], 0) + 1
            most_common = max(complexity_counts, key=complexity_counts.get)
            st.metric("Most Common Complexity", most_common)
        with col4:
            mobile_models = len([m for m in self.models.values() if 'mobile' in m['use_case'].lower() or m['complexity'] in ['Very Low', 'Low']])
            st.metric("Mobile-Ready Models", mobile_models)
        
        # Category distribution chart
        category_data = []
        for category, model_ids in self.categories.items():
            category_data.append({
                'Category': category,
                'Count': len(model_ids),
                'Models': ', '.join([self.models[mid]['name'] for mid in model_ids[:3]]) + ('...' if len(model_ids) > 3 else '')
            })
        
        df_categories = pd.DataFrame(category_data)
        
        fig_categories = px.pie(
            df_categories, 
            values='Count', 
            names='Category',
            title='Model Distribution by Category',
            hover_data=['Models']
        )
        st.plotly_chart(fig_categories, use_container_width=True)
    
    def show_interactive_browser(self) -> None:
        """Interactive model browser with filtering and search"""
        st.subheader("🔍 Interactive Model Browser")
        
        # Search and filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_query = st.text_input(
                "🔎 Search Models",
                placeholder="e.g., mobile, efficient, detection...",
                help="Search by name, description, or use case"
            )
        
        with col2:
            category_filter = st.selectbox(
                "📂 Filter by Category",
                ['All Categories'] + list(self.categories.keys())
            )
        
        with col3:
            complexity_filter = st.selectbox(
                "⚡ Filter by Complexity",
                ['All Complexities', 'Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
            )
        
        # Apply filters
        filtered_models = self._apply_filters(search_query, category_filter, complexity_filter)
        
        st.write(f"**Found {len(filtered_models)} models matching your criteria:**")
        
        # Display filtered models
        if filtered_models:
            self._display_model_grid(filtered_models)
        else:
            st.info("No models match your search criteria. Try adjusting your filters.")
    
    def _apply_filters(self, search_query: str, category_filter: str, complexity_filter: str) -> List[str]:
        """Apply search and filtering criteria"""
        filtered_models = list(self.models.keys())
        
        # Search filter
        if search_query:
            filtered_models = search_engine.search_models(
                search_query,
                category=None if category_filter == 'All Categories' else category_filter,
                complexity=None if complexity_filter == 'All Complexities' else complexity_filter
            )
        
        # Category filter
        if category_filter != 'All Categories':
            category_models = self.categories.get(category_filter, [])
            filtered_models = [m for m in filtered_models if m in category_models]
        
        # Complexity filter
        if complexity_filter != 'All Complexities':
            filtered_models = [
                m for m in filtered_models 
                if self.models[m]['complexity'] == complexity_filter
            ]
        
        return filtered_models
    
    def _display_model_grid(self, model_ids: List[str]) -> None:
        """Display models in a grid layout"""
        
        # Pagination
        models_per_page = 12
        total_pages = (len(model_ids) + models_per_page - 1) // models_per_page
        
        if total_pages > 1:
            page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
        else:
            page = 0
        
        start_idx = page * models_per_page
        end_idx = min(start_idx + models_per_page, len(model_ids))
        current_models = model_ids[start_idx:end_idx]
        
        # Display models in grid (3 columns)
        cols = st.columns(3)
        
        for i, model_id in enumerate(current_models):
            col_idx = i % 3
            info = self.models[model_id]
            
            with cols[col_idx]:
                with st.container():
                    # Model card
                    st.markdown(f"### {info['name']}")
                    st.write(f"**Category:** {info['category']}")
                    st.write(f"**Parameters:** {info['parameters']}")
                    st.write(f"**Complexity:** {info['complexity']}")
                    st.write(f"**Use Case:** {info['use_case'][:50]}...")
                    
                    # Quick stats
                    try:
                        summary = model_manager.get_model_summary(model_id)
                        if summary['memory_mb'] < 50:
                            memory_color = "🟢"
                        elif summary['memory_mb'] < 200:
                            memory_color = "🟡"
                        else:
                            memory_color = "🔴"
                        st.write(f"**Memory:** {memory_color} {summary['memory_mb']:.1f} MB")
                    except:
                        st.write("**Memory:** ⚪ Unknown")
                    
                    # Action buttons
                    if st.button(f"📊 Analyze {info['name']}", key=f"analyze_{model_id}"):
                        st.session_state.selected_model_for_analysis = model_id
                        st.success(f"Selected {info['name']} for analysis!")
                    
                    if st.button(f"🔍 Details", key=f"details_{model_id}"):
                        self._show_model_details_popup(model_id)
    
    def _show_model_details_popup(self, model_id: str) -> None:
        """Show detailed model information in expandable section"""
        info = self.models[model_id]
        
        with st.expander(f"📋 {info['name']} - Detailed Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"• **Full Name:** {info['name']}")
                st.write(f"• **Category:** {info['category']}")
                st.write(f"• **Parameters:** {info['parameters']}")
                st.write(f"• **Complexity:** {info['complexity']}")
                st.write(f"• **Input Shape:** {info['input_shape']}")
                
            with col2:
                st.write("**Usage & Performance:**")
                st.write(f"• **Primary Use Case:** {info['use_case']}")
                st.write(f"• **Description:** {info['description']}")
                
                try:
                    summary = model_manager.get_model_summary(model_id)
                    st.write(f"• **Total Parameters:** {summary['total_parameters']:,}")
                    st.write(f"• **Memory Usage:** {summary['memory_mb']:.1f} MB")
                    
                    if summary['estimated_flops'] > 1e9:
                        flops_str = f"{summary['estimated_flops']/1e9:.1f} GFLOPs"
                    elif summary['estimated_flops'] > 1e6:
                        flops_str = f"{summary['estimated_flops']/1e6:.1f} MFLOPs"
                    else:
                        flops_str = f"{summary['estimated_flops']/1e3:.1f} KFLOPs"
                    st.write(f"• **Computational Cost:** {flops_str}")
                except:
                    st.write("• **Performance Metrics:** Not available")
            
            # Similar models
            similar_models = search_engine.get_similar_models(model_id)
            if similar_models:
                st.write("**Similar Models:**")
                for sim_id, reason, score in similar_models[:3]:
                    sim_info = self.models[sim_id]
                    st.write(f"• **{sim_info['name']}** - {reason} (similarity: {score:.1%})")
    
    def show_comparison_tool(self) -> None:
        """Interactive model comparison tool"""
        st.subheader("⚖️ Model Comparison Tool")
        
        # Model selection for comparison
        st.write("Select up to 4 models to compare:")
        
        selected_models = []
        cols = st.columns(4)
        
        model_options = [f"{info['name']} ({model_id})" for model_id, info in self.models.items()]
        
        for i, col in enumerate(cols):
            with col:
                selection = st.selectbox(
                    f"Model {i+1}",
                    ['None'] + model_options,
                    key=f"compare_model_{i}"
                )
                if selection != 'None':
                    model_id = selection.split('(')[-1].rstrip(')')
                    selected_models.append(model_id)
        
        if len(selected_models) >= 2:
            self._show_comparison_results(selected_models)
    
    def _show_comparison_results(self, model_ids: List[str]) -> None:
        """Display detailed comparison results"""
        st.subheader("📊 Comparison Results")
        
        # Get comparison data
        comparison_data = []
        for model_id in model_ids:
            info = self.models[model_id]
            try:
                summary = model_manager.get_model_summary(model_id)
                comparison_data.append({
                    'Model': info['name'],
                    'Category': info['category'],
                    'Parameters': summary['total_parameters'],
                    'FLOPs': summary['estimated_flops'],
                    'Memory (MB)': summary['memory_mb'],
                    'Complexity': info['complexity'],
                    'Use Case': info['use_case']
                })
            except:
                comparison_data.append({
                    'Model': info['name'],
                    'Category': info['category'],
                    'Parameters': 'Unknown',
                    'FLOPs': 'Unknown',
                    'Memory (MB)': 'Unknown',
                    'Complexity': info['complexity'],
                    'Use Case': info['use_case']
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(df_comparison, use_container_width=True)
            
            # Visualization charts
            numeric_data = []
            for data in comparison_data:
                if isinstance(data['Parameters'], int) and isinstance(data['FLOPs'], int):
                    numeric_data.append(data)
            
            if len(numeric_data) >= 2:
                df_numeric = pd.DataFrame(numeric_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Parameters comparison
                    fig_params = px.bar(
                        df_numeric,
                        x='Model',
                        y='Parameters',
                        title='Parameter Count Comparison',
                        color='Complexity'
                    )
                    st.plotly_chart(fig_params, use_container_width=True)
                
                with col2:
                    # Memory usage comparison
                    fig_memory = px.bar(
                        df_numeric,
                        x='Model',
                        y='Memory (MB)',
                        title='Memory Usage Comparison',
                        color='Category'
                    )
                    st.plotly_chart(fig_memory, use_container_width=True)
                
                # Performance vs Efficiency scatter plot
                fig_scatter = px.scatter(
                    df_numeric,
                    x='Parameters',
                    y='FLOPs',
                    size='Memory (MB)',
                    color='Category',
                    hover_name='Model',
                    title='Performance vs Efficiency Analysis'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    def show_recommendations_engine(self) -> None:
        """AI-powered model recommendation engine"""
        st.subheader("🤖 AI Model Recommendations")
        
        # User requirements input
        st.write("**Tell us about your requirements:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_case = st.text_area(
                "Describe your use case:",
                placeholder="e.g., Real-time object detection for mobile app, High-accuracy image classification for server deployment...",
                height=100
            )
            
            deployment_target = st.selectbox(
                "Deployment Target:",
                ["Mobile/Edge Device", "Server/Cloud", "Embedded System", "Desktop Application", "Not Specified"]
            )
        
        with col2:
            accuracy_priority = st.slider(
                "Accuracy Priority (1=Speed matters more, 10=Accuracy matters more)",
                1, 10, 5
            )
            
            resource_constraints = st.multiselect(
                "Resource Constraints:",
                ["Limited Memory (<100MB)", "Limited CPU", "Limited Power", "Real-time Requirements", "No Constraints"]
            )
        
        if st.button("🚀 Get Recommendations", type="primary"):
            if use_case.strip():
                recommendations = self._generate_smart_recommendations(
                    use_case, deployment_target, accuracy_priority, resource_constraints
                )
                self._display_recommendations(recommendations)
            else:
                st.warning("Please describe your use case to get personalized recommendations.")
    
    def _generate_smart_recommendations(self, use_case: str, deployment_target: str, 
                                      accuracy_priority: int, resource_constraints: List[str]) -> List[Dict]:
        """Generate intelligent model recommendations"""
        
        # Get base recommendations from search engine
        base_recommendations = search_engine.get_recommendations(use_case)
        
        # Score models based on requirements
        scored_models = []
        
        for model_id, base_reason in base_recommendations:
            if model_id not in self.models:
                continue
            
            info = self.models[model_id]
            score = 0
            reasons = [base_reason]
            
            # Deployment target scoring
            if deployment_target == "Mobile/Edge Device":
                if info['complexity'] in ['Very Low', 'Low']:
                    score += 30
                    reasons.append("Optimized for mobile deployment")
                elif info['complexity'] == 'Medium':
                    score += 15
            elif deployment_target == "Server/Cloud":
                if info['complexity'] in ['High', 'Very High']:
                    score += 25
                    reasons.append("High-performance server model")
            
            # Accuracy vs efficiency balance
            complexity_scores = {'Very Low': 1, 'Low': 2, 'Medium': 5, 'High': 8, 'Very High': 9, 'Extreme': 10}
            model_complexity_score = complexity_scores.get(info['complexity'], 5)
            
            if abs(model_complexity_score - accuracy_priority) <= 2:
                score += 20
                reasons.append("Good accuracy-efficiency balance for your needs")
            
            # Resource constraints
            try:
                summary = model_manager.get_model_summary(model_id)
                
                if "Limited Memory (<100MB)" in resource_constraints:
                    if summary['memory_mb'] < 100:
                        score += 25
                        reasons.append("Fits memory constraints")
                    else:
                        score -= 15
                
                if "Real-time Requirements" in resource_constraints:
                    if info['complexity'] in ['Very Low', 'Low', 'Medium']:
                        score += 20
                        reasons.append("Suitable for real-time processing")
                
            except:
                pass
            
            scored_models.append({
                'model_id': model_id,
                'score': score,
                'reasons': reasons,
                'info': info
            })
        
        # Sort by score and return top recommendations
        scored_models.sort(key=lambda x: x['score'], reverse=True)
        return scored_models[:5]
    
    def _display_recommendations(self, recommendations: List[Dict]) -> None:
        """Display recommendation results"""
        st.subheader("💡 Recommended Models")
        
        for i, rec in enumerate(recommendations):
            model_id = rec['model_id']
            info = rec['info']
            score = rec['score']
            reasons = rec['reasons']
            
            # Recommendation ranking
            rank_emoji = ["🥇", "🥈", "🥉", "🏅", "⭐"][i] if i < 5 else "•"
            
            with st.expander(f"{rank_emoji} {info['name']} (Match Score: {score}%)", expanded=i==0):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Category:** {info['category']}")
                    st.write(f"**Parameters:** {info['parameters']}")
                    st.write(f"**Complexity:** {info['complexity']}")
                    st.write(f"**Use Case:** {info['use_case']}")
                    
                    st.write("**Why this model is recommended:**")
                    for reason in reasons:
                        st.write(f"• {reason}")
                
                with col2:
                    try:
                        summary = model_manager.get_model_summary(model_id)
                        st.metric("Memory Usage", f"{summary['memory_mb']:.1f} MB")
                        
                        if summary['estimated_flops'] > 1e9:
                            flops_display = f"{summary['estimated_flops']/1e9:.1f} GFLOPs"
                        else:
                            flops_display = f"{summary['estimated_flops']/1e6:.1f} MFLOPs"
                        st.metric("Computational Cost", flops_display)
                    except:
                        st.write("Performance metrics not available")
                    
                    if st.button(f"Select {info['name']}", key=f"select_rec_{model_id}"):
                        st.session_state.selected_model_for_analysis = model_id
                        st.success(f"Selected {info['name']} for ISA analysis!")


# Global catalog viewer instance
catalog_viewer = ModelCatalogViewer()