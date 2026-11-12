import pytest 

def test_main():
    from src.main import main

    assert main() is None