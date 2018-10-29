""" Cornice services.
"""
from cornice import Service


hello = Service(name='hello', path='/', description="Simplest app")

stretch = Service(name = 'stretch', path='/stretch')

weightloss = Service(name = 'weightloss', path ='/weightloss' )

diet = Service(name = 'diet' , path = 'diet')

@hello.get()
def get_info(request):
    """Returns Hello in JSON."""
    return {'Hello': 'World'}

@stretch.get()
def get_video(request):
    return {}



@weightloss.get()
def get_video(request):
    return {'link': 'https://www.youtube.com/watch?v=LJOtcj9FwNc'}

