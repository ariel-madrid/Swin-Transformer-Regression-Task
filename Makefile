# Nombre del directorio a eliminar
DIRECTORY = swin_base_patch4_window7_224

# Regla por defecto
all: clean

# Regla para eliminar la carpeta
clean:
	@echo "Eliminando el directorio $(DIRECTORY)..."
	@rm -r $(DIRECTORY)
	@echo "Directorio eliminado."
