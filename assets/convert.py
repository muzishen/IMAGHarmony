from pdf2image import convert_from_path

pages = convert_from_path("C:\\Users\\gao\\Downloads\\assets\\harmonybench (1).pdf", dpi=300)
for i, page in enumerate(pages):
    page.save(f"bench.png", "PNG")