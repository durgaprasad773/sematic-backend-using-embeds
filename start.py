"""
Startup script for Syllabus Checker API

This script provides easy commands to start the API server.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def start_api_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server."""
    print(f"🚀 Starting API server on http://{host}:{port}")

    # Ensure data_process directory exists
    os.makedirs("data_process", exist_ok=True)

    try:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
        ]

        if reload:
            cmd.append("--reload")

        print("📋 Server starting with command:")
        print("   " + " ".join(cmd))
        print("\n🌐 API will be available at:")
        print(f"   • Main API: http://{host}:{port}")
        print(f"   • Documentation: http://{host}:{port}/docs")
        print(f"   • Health Check: http://{host}:{port}/health")
        print("\n📂 Available endpoints:")
        print("   • GET  /health - Check system health")
        print("   • POST /similarity-check - Remove similar questions")
        print("   • POST /syllabus-check-text - Filter by syllabus (text)")
        print("   • POST /syllabus-check-file - Filter by syllabus (file)")
        print("   • GET  /download/{filename} - Download processed files")
        print("   • GET  /files - List processed files")
        print("   • POST /cleanup - Clean up processed files")
        print("\n🛑 Press CTRL+C to stop the server")
        print("=" * 60)

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")


def main():
    """Main function to handle startup options."""
    print("🔧 Syllabus Checker API - Startup Script")
    print("=" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "install":
            install_requirements()
            return
        elif command == "test":
            print("🧪 Running API tests...")
            try:
                subprocess.run([sys.executable, "test_api.py"])
            except Exception as e:
                print(f"❌ Test failed: {e}")
            return
        elif command == "help":
            print("📖 Available commands:")
            print("   python start.py install  - Install requirements")
            print("   python start.py test     - Run API tests")
            print("   python start.py start    - Start API server")
            print("   python start.py help     - Show this help")
            return
        elif command != "start":
            print(f"❌ Unknown command: {command}")
            print("   Use 'python start.py help' for available commands")
            return

    # Default: start the server
    # Check if requirements are likely installed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("⚠️  FastAPI dependencies not found!")
        print("   Installing requirements first...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually:")
            print("   pip install -r requirements.txt")
            return

    # Start the server
    start_api_server()


if __name__ == "__main__":
    main()
