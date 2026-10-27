import os
import argparse
import sys
from pathlib import Path

# Add root folder and mini folder to python path to resolve modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "mini"))
from whatsapp_pipeline.pipeline import WhatsAppAIClassifierPipeline

def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Agentic AI Pipeline for Ayurvedic Prakriti Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --chat path/to/chat.txt --media path/to/media/ --output results.xlsx
  python run_pipeline.py --chat chat.txt --media media/ --threshold 0.60 --api-key YOUR_API_KEY
        """
    )
    
    parser.add_argument(
        "--chat",
        required=True,
        help="Path to the exported WhatsApp chat text file (e.g. _chat.txt)"
    )
    
    parser.add_argument(
        "--media",
        required=False,
        help="Path to the directory containing WhatsApp media attachments (optional if --chat is a .zip file)"
    )
    
    parser.add_argument(
        "--output",
        default="whatsapp_prakriti_results.xlsx",
        help="Path to save the generated Excel spreadsheet (default: whatsapp_prakriti_results.xlsx)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Confidence score threshold for forcing predictions (default: 0.55)"
    )
    
    parser.add_argument(
        "--api-key",
        help="Gemini API Key. If not provided, the GEMINI_API_KEY environment variable will be used."
    )
    
    args = parser.parse_args()
    
    # Configure API Key in environment if provided in CLI
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key
        
    # Parse paths
    chat_path = Path(args.chat)
    output_path = Path(args.output)
    
    temp_dir = None
    media_path = None
    
    # Auto-extract if input is a .zip file
    if chat_path.suffix.lower() == ".zip":
        import zipfile
        import tempfile
        
        print(f"Zip file detected. Extracting {chat_path.name}...")
        temp_dir = Path(tempfile.mkdtemp(prefix="whatsapp_pipeline_"))
        
        with zipfile.ZipFile(chat_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Find the chat text file inside the extracted files
        txt_files = list(temp_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError("Could not find any text (.txt) file inside the WhatsApp export zip.")
            
        # Select the text file (often '_chat.txt' or 'WhatsApp Chat with... .txt')
        chat_path = txt_files[0]
        media_path = temp_dir
        print(f"  Extracted chat log: {chat_path.name}")
        print(f"  Extracted media directory: {media_path}")
    else:
        if not args.media:
            parser.error("--media is required when --chat is not a .zip file")
        media_path = Path(args.media)
    
    print("=" * 70)
    print("      Ayurvedic Prakriti WhatsApp Agentic AI Pipeline")
    print("=" * 70)
    print(f"Chat Log:  {chat_path}")
    print(f"Media Dir: {media_path}")
    print(f"Output:    {output_path}")
    print(f"Threshold: {args.threshold:.2f}")
    
    # Check if API Key is loaded
    has_api_key = "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"].strip() != ""
    if has_api_key:
        print("API Status: Live Gemini API Mode (Contextual LLM + Multimodal CV)")
    else:
        print("API Status: Local Fallback Mode (Token heuristics + chronological view ordering)")
    print("=" * 70)
    
    try:
        pipeline = WhatsAppAIClassifierPipeline(confidence_threshold=args.threshold)
        results = pipeline.process_chat_export(
            chat_txt_path=chat_path,
            media_dir_path=media_path,
            output_excel_path=output_path
        )
        
        # Append successfully processed subjects to the main dataset Excel + Wrist_Dataset
        from whatsapp_pipeline.exporter import append_to_source_dataset
        append_to_source_dataset(
            results,
            xlsx_path=Path("dataset/palm_dataset_26 June 26.xlsx"),
            wrist_dataset_dir=Path("dataset/Wrist_Dataset")
        )
        
        # Summary statistics
        total = len(results)
        flagged = sum(1 for r in results if r["needs_review"])
        successful = total - flagged
        
        print("\n" + "=" * 70)
        print("                          Execution Summary")
        print("=" * 70)
        print(f"Total Subjects Processed:      {total}")
        print(f"Successfully Analyzed:        {successful}")
        print(f"Flagged for Manual Review:    {flagged}")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if temp_dir and temp_dir.exists():
            import shutil
            print(f"Cleaning up temporary directory: {temp_dir.name}...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    sys.exit(main())
