import requests
import json

def test_trading212_api(api_key, is_demo=False):
    base_url = "https://demo.trading212.com/api/v0" if is_demo else "https://live.trading212.com/api/v0"
    headers = {"Authorization": api_key}
    
    print(f"Testing API endpoint: {base_url}")
    
    try:
        # Test cash balance endpoint
        print("\nTesting /equity/account/cash...")
        response = requests.get(
            f"{base_url}/equity/account/cash",
            headers=headers,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print("Response Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")
            
        print("\nResponse Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <api_key> [--demo]")
        print("  --demo   Use demo API endpoint")
        sys.exit(1)
        
    api_key = sys.argv[1]
    is_demo = "--demo" in sys.argv
    
    test_trading212_api(api_key, is_demo)
