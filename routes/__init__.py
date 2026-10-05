def register_blueprints(app):
    from .audio import audio_bp
    from .video import video_bp
    from .image import image_bp
    from .assistant import assistant_bp

    app.register_blueprint(audio_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(assistant_bp)
