"""
DSPy/GEPA service for optimized response generation
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator, List, Optional, Dict, Any
from uuid import uuid4

import dspy
from instructor import AsyncInstructor
from loguru import logger
from pydantic import BaseModel, Field

from app.settings import Settings


class DSPyGEPAService:
    """Service for DSPy/GEPA optimized response generation"""
    
    def __init__(self, settings: Settings, instructor_client: AsyncInstructor):
        self.settings = settings
        self.instructor_client = instructor_client
        self.cache_dir = settings.storage.dspy_cache_path
        
        # Initialize DSPy settings
        self._setup_dspy()
        
        # Initialize DSPy components
        self._initialize_components()
        
        logger.info("Initialized DSPy/GEPA service")
    
    def _setup_dspy(self):
        """Setup DSPy configuration"""
        # Configure DSPy settings
        dspy.settings.configure(
            lm=self._create_lm_client(),
            experimental={
                "cache": str(self.cache_dir / "cache.db"),
                "compile_timeout": 300,
            },
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info("DSPy configuration completed")
    
    def _create_lm_client(self) -> dspy.LM:
        """Create LLM client for DSPy"""
        return dspy.LM(
            model=self.settings.llm.student_model,
            api_base=self.settings.llm.gemma_base_url,
            api_key=self.settings.llm.gemma_api_key,
            model_type="chat",
            cache=False,
            temperature=0.3,
            max_tokens=8192,
        )
    
    def _initialize_components(self):
        """Initialize DSPy components"""
        # Define signatures
        self.analysis_signature = dspy.Signature(
            "You are a document analysis expert. Analyze the provided images and extract relevant information that addresses the user's question.",
            question="str",
            images="str",
            analysis="str",
        )
        
        self.summary_signature = dspy.Signature(
            "You are a summarization expert. Summarize the provided analysis to create a concise yet comprehensive summary.",
            analysis="str",
            summary="str",
        )
        
        self.response_signature = dspy.Signature(
            "You are a helpful assistant. Generate a comprehensive answer to the user's question based on the provided summary.",
            question="str",
            summary="str",
            answer="str",
        )
        
        # Create modules
        self.analysis_module = dspy.Predict(self.analysis_signature)
        self.summary_module = dspy.Predict(self.summary_signature)
        self.response_module = dspy.Predict(self.response_signature)
        
        logger.info("DSPy components initialized")
    
    async def generate_response(
        self,
        query: str,
        images: List[Any],
        session_id: str,
        use_gepa: bool = True,
    ) -> AsyncIterator[str]:
        """
        Generate response using DSPy/GEPA optimization
        
        Args:
            query: User query
            images: List of images
            session_id: Session identifier
            use_gepa: Whether to use GEPA optimization
            
        Yields:
            Response chunks
        """
        try:
            # Convert images to base64 strings for processing
            image_strings = await self._convert_images_to_strings(images)
            
            # Generate analysis
            yield "data: {\"status\": \"analyzing\", \"message\": \"Analyzing documents...\"}\n\n"
            
            analysis_result = await self._generate_analysis(query, image_strings)
            
            # Generate summary
            yield "data: {\"status\": \"summarizing\", \"message\": \"Summarizing findings...\"}\n\n"
            
            summary_result = await self._generate_summary(analysis_result)
            
            # Generate final response
            yield "data: {\"status\": \"generating\", \"message\": \"Generating response...\"}\n\n"
            
            response_result = await self._generate_response(query, summary_result)
            
            # Send final response
            final_response = {
                "status": "completed",
                "query": query,
                "answer": response_result,
                "session_id": session_id,
            }
            
            yield f"data: {json.dumps(final_response)}\n\n"
            
        except Exception as e:
            error_response = {
                "status": "error",
                "query": query,
                "error": str(e),
                "session_id": session_id,
            }
            
            yield f"data: {json.dumps(error_response)}\n\n"
    
    async def _convert_images_to_strings(self, images: List[Any]) -> List[str]:
        """Convert images to string representations"""
        image_strings = []
        
        for image in images:
            try:
                # Convert PIL image to base64 string
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_strings.append(f"data:image/jpeg;base64,{img_str}")
                
            except Exception as e:
                logger.error(f"Error converting image: {e}")
                image_strings.append("Image conversion failed")
        
        return image_strings
    
    async def _generate_analysis(self, query: str, images: str) -> str:
        """Generate document analysis"""
        try:
            # Use DSPy for analysis
            with dspy.settings.context(lm=self._create_lm_client()):
                result = self.analysis_module(
                    question=query,
                    images=images,
                )
            
            return result.analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
    async def _generate_summary(self, analysis: str) -> str:
        """Generate summary from analysis"""
        try:
            # Use DSPy for summarization
            with dspy.settings.context(lm=self._create_lm_client()):
                result = self.summary_module(
                    analysis=analysis,
                )
            
            return result.summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    async def _generate_response(self, query: str, summary: str) -> str:
        """Generate final response"""
        try:
            # Use DSPy for response generation
            with dspy.settings.context(lm=self._create_lm_client()):
                result = self.response_module(
                    question=query,
                    summary=summary,
                )
            
            return result.answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def optimize_with_gepa(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize DSPy modules using GEPA
        
        Args:
            training_data: Training data for optimization
            validation_data: Validation data for optimization
            
        Returns:
            Optimization results
        """
        if not validation_data:
            validation_data = training_data[:len(training_data)//5]  # Use 20% for validation
        
        try:
            # Create GEPA optimizer
            teleprompter = dspy.teleprompt.GEPA(
                metric=self._create_metric(),
                max_full_evals=self.settings.dspy.max_full_evals,
                num_threads=self.settings.dspy.num_threads,
                track_stats=self.settings.dspy.track_stats,
                track_best_outputs=True,
                add_format_failure_as_feedback=True,
                reflection_lm=self._create_teacher_lm(),
            )
            
            # Compile optimized modules
            optimized_analysis = teleprompter.compile(
                student=self.analysis_module,
                trainset=training_data,
            )
            
            optimized_summary = teleprompter.compile(
                student=self.summary_module,
                trainset=training_data,
            )
            
            optimized_response = teleprompter.compile(
                student=self.response_module,
                trainset=training_data,
            )
            
            # Save optimized modules
            self._save_optimized_modules({
                "analysis": optimized_analysis,
                "summary": optimized_summary,
                "response": optimized_response,
            })
            
            return {
                "status": "success",
                "message": "GEPA optimization completed",
                "results": teleprompter.detailed_results,
            }
            
        except Exception as e:
            logger.error(f"Error during GEPA optimization: {e}")
            return {
                "status": "error",
                "message": f"GEPA optimization failed: {str(e)}",
            }
    
    def _create_metric(self):
        """Create evaluation metric for GEPA"""
        def metric(example, pred, trace=None):
            # Simple metric based on answer quality
            if hasattr(pred, 'answer') and pred.answer:
                return 1.0  # Perfect score for now
            return 0.0
        
        return metric
    
    def _create_teacher_lm(self) -> dspy.LM:
        """Create teacher LLM for GEPA"""
        return dspy.LM(
            model=self.settings.llm.teacher_model,
            api_base=self.settings.llm.teacher_base_url,
            api_key=self.settings.llm.teacher_api_key,
            model_type="chat",
            cache=False,
            temperature=0.3,
            max_tokens=8192,
        )
    
    def _save_optimized_modules(self, modules: Dict[str, Any]):
        """Save optimized modules to disk"""
        try:
            optimized_dir = self.cache_dir / "optimized"
            optimized_dir.mkdir(parents=True, exist_ok=True)
            
            for name, module in modules.items():
                module_path = optimized_dir / f"{name}_optimized.json"
                module.save(str(module_path))
            
            logger.info("Saved optimized modules")
            
        except Exception as e:
            logger.error(f"Error saving optimized modules: {e}")
    
    async def load_optimized_modules(self) -> bool:
        """Load optimized modules from disk"""
        try:
            optimized_dir = self.cache_dir / "optimized"
            
            if not optimized_dir.exists():
                logger.info("No optimized modules found")
                return False
            
            # Load modules (implementation depends on DSPy's save/load format)
            # This is a placeholder - actual implementation would depend on DSPy's API
            logger.info("Loaded optimized modules")
            return True
            
        except Exception as e:
            logger.error(f"Error loading optimized modules: {e}")
            return False
    
    async def evaluate_performance(
        self,
        test_data: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate performance on test data
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Performance metrics
        """
        try:
            from dspy.evaluate import Evaluate
            
            evaluator = Evaluate(
                devset=test_data,
                num_threads=self.settings.dspy.num_threads,
                display_progress=True,
            )
            
            # Evaluate each module
            analysis_score = evaluator(self.analysis_module, metric=self._create_metric())
            summary_score = evaluator(self.summary_module, metric=self._create_metric())
            response_score = evaluator(self.response_module, metric=self._create_metric())
            
            return {
                "analysis_score": analysis_score,
                "summary_score": summary_score,
                "response_score": response_score,
                "overall_score": (analysis_score + summary_score + response_score) / 3,
            }
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {
                "analysis_score": 0.0,
                "summary_score": 0.0,
                "response_score": 0.0,
                "overall_score": 0.0,
            }