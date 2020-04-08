from mne_nirs import foo


def test_foo():
    print(123456)
    assert foo() == 'foo'
