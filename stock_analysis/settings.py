INSTALLED_APPS = [
    # ...
    'stocks',  # Ensure your app is listed here
    # ...
]

# Database settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # or your preferred database
        'NAME': BASE_DIR / "db.sqlite3",  # Adjust as necessary
    }
}

# Template settings
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'stocks/templates')],  # Ensure this path is correct
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                # ...
            ],
        },
    },
]

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'stocks/static'),  # Adjust as necessary
] 