smart_home_data = [
    {
        "id": "light_001",
        "name": "Smart LED Bulb",
        "category": "Lighting",
        "description": "A WiFi-enabled LED bulb with adjustable brightness and color temperature.",
        "features": "Dimmable, RGB colors, voice control, energy efficient",
        "url": "example.com/light_001",
        "price": "2.00 AED",
    },
    {
        "id": "thermo_001",
        "name": "Smart Thermostat",
        "category": "Climate",
        "description": "Learns user preferences and automatically adjusts room temperature.",
        "features": "AI scheduling, remote control, energy saving mode",
        "url": "example.com/thermo_001",
        "price": "20.00 AED",
    },
    {
        "id": "lock_001",
        "name": "Smart Door Lock",
        "category": "Security",
        "description": "Keyless entry system controlled via mobile app or fingerprint.",
        "features": "Fingerprint, PIN, remote unlock, auto-lock",
        "url": "example.com/lock_001",
        "price": "100.00 AED",
    },
    {
        "id": "smoke_001",
        "name": "Smart Smoke Detector",
        "category": "Safety",
        "description": "Wi-Fi enabled smoke detector with real-time alerts and voice alarms.",
        "features": "Smoke detection, mobile alerts, battery backup, voice warning",
        "url": "example.com/smoke_001",
        "price": "150.00 AED"
    },
    {
        "id": "cam_001",
        "name": "Smart Security Camera",
        "category": "Security",
        "description": "Indoor/outdoor camera with motion detection and night vision.",
        "features": "1080p, motion alerts, cloud storage, night vision",
        "url": "example.com/cam_001",
        "price": "200.00 AED",
    },
    {
        "id": "cam_001",
        "name": "Smart Smoke detector",
        "category": "Security",
        "description": "Indoor/outdoor camera with motion detection and night vision.",
        "features": "1080p, motion alerts, cloud storage, night vision",
        "url": "example.com/cam_001",
        "price": "200.00 AED",
    },
    {
        "id": "plug_001",
        "name": "Smart Plug",
        "category": "Energy",
        "description": "Turns any appliance into a smart device with remote control.",
        "features": "App control, scheduling, energy monitoring",
        "url": "example.com/plug_001",
        "price": "25 AED",
    },
    {
        "id": "speaker_001",
        "name": "Smart Voice Speaker",
        "category": "Entertainment",
        "description": "A voice-controlled smart speaker with integrated virtual assistant and multi-room audio support.",
        "features": "Voice assistant, Bluetooth, WiFi streaming, multi-room sync",
        "url": "example.com/speaker_001",
        "price": "45.00 AED"
    },
    {
        "id": "tv_001",
        "name": "Smart 4K TV",
        "category": "Entertainment",
        "description": "Ultra HD smart television with built-in streaming apps and voice assistant support.",
        "features": "4K UHD, HDR, voice control, screen casting, WiFi connectivity",
        "url": "example.com/tv_001",
        "price": "550.00 AED"
    },
    {
        "id": "vacuum_001",
        "name": "Smart Robot Vacuum",
        "category": "Cleaning",
        "description": "Autonomous vacuum cleaner with intelligent room mapping and obstacle avoidance.",
        "features": "LiDAR navigation, scheduled cleaning, auto recharge, app control",
        "url": "example.com/vacuum_001",
        "price": "180.00 AED"
    },
    {
        "id": "fridge_001",
        "name": "Smart Refrigerator",
        "category": "Kitchen",
        "description": "WiFi-enabled refrigerator with inventory tracking and touchscreen controls.",
        "features": "Internal camera, grocery tracking, touch display, temperature alerts",
        "url": "example.com/fridge_001",
        "price": "950.00 AED"
    },
    {
        "id": "mirror_001",
        "name": "Smart Fitness Mirror",
        "category": "Health",
        "description": "Interactive fitness mirror that streams workouts and tracks exercise performance.",
        "features": "Live workouts, AI posture correction, heart rate sync, voice commands",
        "url": "example.com/mirror_001",
        "price": "650.00 AED"
    },
    {
        "id": "garden_001",
        "name": "Smart Garden Hub",
        "category": "Outdoor",
        "description": "Automated plant monitoring and irrigation system for indoor and outdoor gardens.",
        "features": "Moisture sensors, automatic watering, weather sync, mobile alerts",
        "url": "example.com/garden_001",
        "price": "75.00 AED"
    }


]

user_data = [
    {
        "id": "1",
        "name": "Ahmed",
        "email": "ahmed@example.com",
        "elife_member": True,
        "crm": "postpaid",
        "phone": "123456789"
    },
    {
        "id": "2",
        "name": "Mohamed",
        "email": "mohamed@example.com",
        "crm": "postpaid",
        "elife_member": False,
        "phone": "987654321"
    },
    {
        "id": "3",
        "name": "Eyad",
        "email": "eyad@example.com",
        "elife_member": False,
        "crm": "prepaid",
        "phone": "123456789"
    },
]

def get_user_by_email(email: str):
    """Retrieve user data by email."""
    for user in user_data:
        if user["email"].lower() == email.lower():
            return user
    return None
