"""Application entry point for the 财报博弈 Dash 仪表盘."""

from app_core import create_dash_app

app = create_dash_app()


if __name__ == "__main__":
    app.run_server(debug=True)
