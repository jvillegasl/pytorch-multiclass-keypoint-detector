test:
	pytest --disable-warnings -v $(if $(c),-k $(c),)
# ifndef c
# 	# python -m unittest __tests__
# 	pytest --disable-warnings -v
# else
# 	# python -m unittest __tests__.test_$(c)
# 	pytest --disable-warnings -v -k $(c)
# endif