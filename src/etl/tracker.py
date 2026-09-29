"""
Módulo de Seguimiento y Auditoría del Pipeline (MLOps Tracker).
Registra métricas, duraciones y estados en la tabla 'pipeline_ejecuciones'.
"""
import json
import logging
import traceback
from datetime import datetime
from src.db.connection import get_connection

logger = logging.getLogger(__name__)

class PipelineTracker:
    def __init__(self, tipo_ejecucion: str, ejecutado_por: str = "CLI_LOCAL"):
        self.tipo_ejecucion = tipo_ejecucion
        self.ejecutado_por = ejecutado_por
        self.ejecucion_id = None
        self.iniciado_en = None
        self.conn = None
        self._exito = False

    def __enter__(self):
        self.iniciado_en = datetime.now()
        try:
            self.conn = get_connection()
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pipeline_ejecuciones (
                        tipo_ejecucion, ejecutado_por, iniciado_en, estado
                    ) VALUES (%s, %s, %s, 'EN_PROCESO')
                    RETURNING id;
                """, (self.tipo_ejecucion, self.ejecutado_por, self.iniciado_en))
                self.ejecucion_id = cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"[{self.tipo_ejecucion}] Corrida #{self.ejecucion_id} iniciada a las {self.iniciado_en.isoformat()}.")
        except Exception as e:
            logger.warning(f"No se pudo registrar inicio en pipeline_ejecuciones: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and not self._exito:
            finalizado_en = datetime.now()
            duracion = int((finalizado_en - self.iniciado_en).total_seconds())
            error_trace = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            logger.error(f"[{self.tipo_ejecucion}] Falló la corrida #{self.ejecucion_id}: {exc_val}")
            
            if self.conn and not self.conn.closed:
                try:
                    with self.conn.cursor() as cur:
                        cur.execute("""
                            UPDATE pipeline_ejecuciones
                            SET finalizado_en = %s,
                                duracion_segundos = %s,
                                errores = 1,
                                observaciones = %s,
                                estado = 'FALLIDO'
                            WHERE id = %s;
                        """, (finalizado_en, duracion, error_trace, self.ejecucion_id))
                    self.conn.commit()
                except Exception as db_err:
                    logger.warning(f"No se pudo registrar fallo en BD: {db_err}")
                finally:
                    self.conn.close()

    def completar_exito(self, leidos: int, guardados: int, filtrados: int = 0, 
                        metricas: dict = None, observaciones: str = None):
        """Registra la culminación exitosa de la corrida."""
        self._exito = True
        finalizado_en = datetime.now()
        duracion = int((finalizado_en - self.iniciado_en).total_seconds())
        metricas_json = json.dumps(metricas) if metricas else None
        
        logger.info(
            f"[{self.tipo_ejecucion}] Corrida #{self.ejecucion_id} COMPLETADA en {duracion}s. "
            f"Leídos: {leidos}, Guardados: {guardados}, Filtrados: {filtrados}."
        )
        
        if self.conn and not self.conn.closed:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        UPDATE pipeline_ejecuciones
                        SET finalizado_en = %s,
                            duracion_segundos = %s,
                            registros_leidos = %s,
                            registros_guardados = %s,
                            registros_filtrados = %s,
                            metricas_salida = %s,
                            observaciones = %s,
                            estado = 'COMPLETADO'
                        WHERE id = %s;
                    """, (finalizado_en, duracion, leidos, guardados, filtrados, metricas_json, observaciones, self.ejecucion_id))
                self.conn.commit()
            except Exception as e:
                logger.warning(f"No se pudo actualizar estado COMPLETADO en BD: {e}")
            finally:
                self.conn.close()
