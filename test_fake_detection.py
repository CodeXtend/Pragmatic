"""
Test script to verify fake content detection.
Tests the NEWS-BASED verification system.
Checks if content is covered in Google News and verifies against real news sources.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from tools.google_factcheck_tool import GoogleFactCheckTool, verify_video_content
from tools.news_verification_tool import NewsVerificationTool, verify_content_with_news

# Test content - the skateboarding in front of bus scenario
test_content = """
This frame captures a bustling urban street scene during daytime, with a man walking across the road in the foreground while various vehicles populate the lanes. A yellow auto-rickshaw, a white and red bus, and a colorful yellow-orange truck are prominent, alongside cars and a motorcycle, all framed by an elevated concrete structure on the left and lush green trees and buildings in the background. The bright sunlight illuminates the active flow of traffic and pedestrians, creating a dynamic city atmosphere. A person walks down the middle of a bustling multi-lane urban road, observed from an elevated perspective, with a white and red bus, a black SUV, a motorcycle, and a red truck navigating the traffic around them. Lush green trees and various buildings line the background, while a large concrete elevated structure frames the left side, depicting a busy daytime city environment. A person is skateboarding on their stomach down the center of a busy urban road, directly in front of a large white and red bus, while various other vehicles like an SUV, a colorful truck, and a motorcycle navigate the surrounding lanes. The scene is set on a multi-lane street lined with trees and an overhead concrete flyover, conveying a chaotic yet daring atmosphere amidst regular city traffic. A busy urban street scene unfolds under bright daylight, featuring a white and red bus prominently moving towards the viewer, followed by a black car and a yellow auto-rickshaw, while a colorful, decorated truck proceeds in the opposite direction. The multi-lane road is flanked by lush green trees, with an elevated concrete structure on the left and a small temple with ornate spires visible on the right, suggesting a bustling daily atmosphere. Various other vehicles, including a scooter and another auto-rickshaw, along with a few pedestrians, contribute to the active, typical city environment. Check this above incident is fake or real by the real source.
"""

# Additional test cases
test_cases = [
    {
        "name": "Viral Skateboard Stunt (Should be FAKE - no news coverage)",
        "content": test_content,
        "expected_fake": True
    },
    {
        "name": "Real News Event Test",
        "content": "Prime Minister Narendra Modi addressed the nation on Independence Day 2024 from Red Fort Delhi",
        "expected_fake": False
    },
    {
        "name": "Fake Viral Video Pattern",
        "content": "Person lying on railway tracks as train approaches, filmed from above, viral video",
        "expected_fake": True
    }
]

def test_news_verification():
    """Test the news-based verification system."""
    print("="*70)
    print("🧪 NEWS-BASED VERIFICATION TEST")
    print("="*70)
    
    # Initialize the news verification tool
    news_tool = NewsVerificationTool()
    
    print("\n📝 Testing with skateboarding-in-front-of-bus content...\n")
    
    # Run news-based verification
    result = news_tool.comprehensive_verify(test_content, verbose=True)
    
    print("\n" + "="*70)
    print("📊 NEWS VERIFICATION RESULTS")
    print("="*70)
    print(f"Verdict: {result.verdict}")
    print(f"Is Fake: {result.is_fake}")
    print(f"Is Verified in News: {result.is_verified}")
    print(f"Confidence: {result.confidence*100:.1f}%")
    print(f"Action: {result.action}")
    print(f"Summary: {result.verdict_summary}")
    print(f"Detection Method: {result.detection_method}")
    print(f"News Sources Found: {len(result.news_sources)}")
    if result.extracted_incident:
        print(f"Incident Extracted: {result.extracted_incident.main_event}")
    print("="*70)
    
    return result

def test_comprehensive_check():
    """Test the comprehensive fake check (includes news verification)."""
    print("\n\n" + "="*70)
    print("🧪 COMPREHENSIVE FAKE CHECK TEST")
    print("="*70)
    
    tool = GoogleFactCheckTool()
    
    result = tool.comprehensive_fake_check(test_content, verbose=True)
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE CHECK RESULTS")
    print("="*70)
    print(f"Is Fake: {result.is_fake}")
    print(f"Confidence: {result.confidence*100:.1f}%")
    print(f"Action: {result.action}")
    print(f"Verdict: {result.verdict_summary}")
    print(f"Detection Method: {result.detection_method}")
    print(f"Sources: {result.sources}")
    print("="*70)
    
    if result.is_fake:
        print("\n✅ TEST PASSED - System correctly identified content as FAKE")
    else:
        print("\n❌ TEST FAILED - System did not identify content as FAKE")
    
    return result.is_fake

def test_multiple_cases():
    """Test multiple content cases."""
    print("\n\n" + "="*70)
    print("🧪 MULTIPLE TEST CASES")
    print("="*70)
    
    news_tool = NewsVerificationTool()
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print("="*70)
        
        result = news_tool.comprehensive_verify(test_case['content'], verbose=True)
        
        passed = result.is_fake == test_case['expected_fake']
        results.append({
            "name": test_case['name'],
            "expected": test_case['expected_fake'],
            "actual": result.is_fake,
            "passed": passed,
            "verdict": result.verdict
        })
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n{status} - Expected: {test_case['expected_fake']}, Got: {result.is_fake}")
    
    # Summary
    print("\n\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    passed_count = sum(1 for r in results if r['passed'])
    print(f"Passed: {passed_count}/{len(results)}")
    for r in results:
        status = "✅" if r['passed'] else "❌"
        print(f"  {status} {r['name']}: {r['verdict']}")
    
    return passed_count == len(results)

def main():
    print("="*70)
    print("🧪 FAKE CONTENT DETECTION TEST - NEWS-BASED VERIFICATION")
    print("="*70)
    print("\nThis test verifies content against REAL NEWS SOURCES")
    print("If an incident is not covered by news media, it's likely FAKE/STAGED\n")
    
    # Test 1: News Verification Only
    print("\n" + "="*70)
    print("📋 TEST 1: News-Based Verification")
    print("="*70)
    news_result = test_news_verification()
    
    # Test 2: Comprehensive Check
    print("\n" + "="*70)
    print("📋 TEST 2: Comprehensive Fake Check")
    print("="*70)
    comprehensive_passed = test_comprehensive_check()
    
    # Final result
    print("\n\n" + "="*70)
    print("📊 FINAL TEST RESULT")
    print("="*70)
    
    if news_result.is_fake or comprehensive_passed:
        print("\n✅ SUCCESS - The news-based verification system is working!")
        print("   - Content not found in legitimate news sources")
        print("   - Correctly identified as FAKE/STAGED content")
        return True
    else:
        print("\n⚠️ System needs adjustment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
