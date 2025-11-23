raw_pdf_text="""
lorem798 ipsum dolor sit amet, 

consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut 
labore et dolore magna aliqua. Ut enim ad minim veniam, qu
is nostrud exercitation ullamco laboris nisi ut aliquip ex ea 
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.            
"""

#cleaning function
def clean_text(text):
    # Remove leading/trailing whitespace
    text= " ".join(text.split())
   
   #fix ligatures
    text=text.replace("ﬁ","fi")
    text=text.replace("ﬂ","fl")

    return text

cleaned_text=clean_text(raw_pdf_text)
print("Before")
print(repr(raw_pdf_text[:100]))
print("After")
print(repr(cleaned_text[:100]))


