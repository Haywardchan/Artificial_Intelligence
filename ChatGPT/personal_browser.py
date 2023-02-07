import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Browser(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a widget for the browser window
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Create the navigation bar
        self.nav_bar = QLineEdit(self)
        self.nav_bar.returnPressed.connect(self.navigate_to_url)

        # Create the web view
        self.web_view = QWebEngineView(self)
        self.web_view.load(QUrl("http://www.google.com"))

        # Create the back button
        self.back_button = QPushButton("<", self)
        self.back_button.clicked.connect(self.web_view.back)

        # Create the forward button
        self.forward_button = QPushButton(">", self)
        self.forward_button.clicked.connect(self.web_view.forward)

        # Create the refresh button
        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self.web_view.reload)

        # Set up the layout
        layout = QVBoxLayout(self.main_widget)
        layout.addWidget(self.nav_bar)
        layout.addWidget(self.web_view)
        layout.addWidget(self.back_button)
        layout.addWidget(self.forward_button)
        layout.addWidget(self.refresh_button)

        # Set the window properties
        self.setWindowTitle("Browser")
        self.setGeometry(100, 100, 800, 600)

    def navigate_to_url(self):
        url = self.nav_bar.text()
        if not url.startswith("http"):
            url = "http://" + url
        self.web_view.load(QUrl(url))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    browser = Browser()
    browser.show()
    sys.exit(app.exec_())
