from utils.preprocess import preprocess

def test1():
    output = preprocess("this is my test case!")
    assert output == "test case"

def test2():
    output = preprocess("doctors restore ken burns' full-color vision after removing massive tumor from filmmaker's visual cortex")
    assert output == "doctor restore ken burn full color vision remove massive tumor filmmaker visual cortex"

if __name__ == "__main__":
    test1()
    test2()
    print("All tests complete.")