"""
One-time script to save Instagram session.
Run this script ONCE as administrator to save your session.

Steps:
1. Right-click on PowerShell/Command Prompt -> Run as Administrator
2. Navigate to this folder
3. Run: python save_session.py
4. Enter your credentials when prompted
5. After success, run reel_scrapper.py normally (no admin needed)
"""

import instaloader

INSTAGRAM_USERNAME = "192.168.0.29"

L = instaloader.Instaloader()

print("=" * 50)
print("Instagram Session Saver")
print("=" * 50)
print(f"\nUsername: {INSTAGRAM_USERNAME}")

password = input("Enter your Instagram password: ")

try:
    print("\nLogging in...")
    L.login(INSTAGRAM_USERNAME, password)
    L.save_session_to_file()
    print("\n✅ SUCCESS! Session saved.")
    print("You can now run reel_scrapper.py without admin privileges.")
except instaloader.exceptions.TwoFactorAuthRequiredException:
    print("\n⚠️ Two-Factor Authentication required!")
    code = input("Enter 2FA code from your authenticator app: ")
    try:
        L.two_factor_login(code)
        L.save_session_to_file()
        print("\n✅ SUCCESS! Session saved with 2FA.")
    except Exception as e:
        print(f"\n❌ 2FA failed: {e}")
except instaloader.exceptions.BadCredentialsException:
    print("\n❌ Wrong password! Please check and try again.")
except instaloader.exceptions.ConnectionException as e:
    print(f"\n❌ Connection error: {e}")
except Exception as e:
    print(f"\n❌ Login failed: {e}")
    print("\nIf you're getting rate-limited, wait a few minutes and try again.")
