#!/usr/bin/env python3
"""
Setup script for Google Drive integration with SAT Question Generator.
"""

import os
import sys
import json
import argparse
from src.utils import setup_gdrive_credentials

def main():
    """
    Run the setup process for Google Drive integration.
    """
    parser = argparse.ArgumentParser(description="Set up Google Drive credentials for SAT Question Generator.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--file', '-f', type=str, help="Path to a Google Drive credentials JSON file")
    group.add_argument('--json', '-j', type=str, help="Google Drive credentials as a JSON string")
    group.add_argument('--new', '-n', action='store_true', help="Create a new client_secrets.json file for authentication")
    
    parser.add_argument('--output', '-o', type=str, help="Output path for credentials file", 
                       default=os.getenv("GDRIVE_CREDENTIALS_PATH", "gdrive_credentials.json"))
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Google Drive Setup for SAT Question Generator".center(80))
    print("=" * 80 + "\n")
    
    # Handle file input
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                credentials_data = f.read()
            
            success = setup_gdrive_credentials(credentials_data, args.output)
            if success:
                print("\nGoogle Drive credentials successfully set up from file.")
                print(f"Credentials file path: {args.output}")
                print("\nThis path will be used by default, or you can set the GDRIVE_CREDENTIALS_PATH environment variable.")
                return 0
            else:
                print("\nFailed to set up Google Drive credentials from file.")
                return 1
                
        except Exception as e:
            print(f"\nError reading credentials file: {str(e)}")
            return 1
    
    # Handle JSON string input
    elif args.json:
        success = setup_gdrive_credentials(args.json, args.output)
        if success:
            print("\nGoogle Drive credentials successfully set up from JSON string.")
            print(f"Credentials file path: {args.output}")
            print("\nThis path will be used by default, or you can set the GDRIVE_CREDENTIALS_PATH environment variable.")
            return 0
        else:
            print("\nFailed to set up Google Drive credentials from JSON string.")
            return 1
    
    # Handle new credentials setup
    elif args.new:
        try:
            from pydrive.auth import GoogleAuth
            
            # Create client_secrets.json template
            client_secrets = {
                "installed": {
                    "client_id": "YOUR_CLIENT_ID",
                    "client_secret": "YOUR_CLIENT_SECRET",
                    "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://accounts.google.com/o/oauth2/token"
                }
            }
            
            with open("client_secrets.json", "w", encoding="utf-8") as f:
                json.dump(client_secrets, f, indent=4)
            
            print("\nCreated template client_secrets.json file.")
            print("Please edit this file with your Google API credentials from the Google Developer Console.")
            print("Visit: https://console.developers.google.com/ to create credentials.")
            print("\nAfter editing, run the following command:")
            print("python setup_gdrive.py")
            
            return 0
            
        except ImportError:
            print("\nError: Required package 'pydrive' not installed.")
            print("Please install it with: pip install pydrive")
            return 1
        except Exception as e:
            print(f"\nError creating client_secrets.json template: {str(e)}")
            return 1
    
    # Default: try to authenticate using existing client_secrets.json
    else:
        try:
            from pydrive.auth import GoogleAuth
            
            if not os.path.exists("client_secrets.json"):
                print("\nNo client_secrets.json file found.")
                print("Please run again with --new to create a template, or provide credentials file/JSON.")
                return 1
                
            # Try to authenticate
            print("\nAttempting to authenticate with Google Drive...")
            gauth = GoogleAuth()
            
            # Force a refresh token
            gauth.LocalWebserverAuth()
            
            # Save the credentials file
            gauth.SaveCredentialsFile("credentials.txt")
            
            print("\nAuthentication successful.")
            print("Credentials saved to 'credentials.txt'.")
            print("\nYou can now use the SAT Question Generator with Google Drive integration.")
            
            return 0
            
        except ImportError:
            print("\nError: Required package 'pydrive' not installed.")
            print("Please install it with: pip install pydrive")
            return 1
        except Exception as e:
            print(f"\nError during authentication: {str(e)}")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 