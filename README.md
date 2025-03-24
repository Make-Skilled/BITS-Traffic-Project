# Traffic Signal Analysis System

A Flask-based web application for intelligent traffic signal management that analyzes traffic density across four directions and dynamically controls signal timing to optimize traffic flow.

## Features

- User authentication (signup/login)
- Real-time traffic density monitoring
- Dynamic signal timing control
- Traffic flow statistics and analytics
- Modern UI with Tailwind CSS
- Responsive dashboard

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd traffic-signal-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///traffic_system.db
```

5. Initialize the database:
```bash
flask shell
>>> from app import db
>>> db.create_all()
>>> exit()
```

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
traffic-signal-system/
├── app.py              # Main application file
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── base.html      # Base template
│   ├── index.html     # Landing page
│   ├── login.html     # Login page
│   ├── signup.html    # Signup page
│   └── dashboard.html # Main dashboard
└── static/            # Static files (CSS, JS, images)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 