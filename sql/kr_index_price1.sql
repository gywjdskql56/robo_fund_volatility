select b.TD AS DATE
, b.SCODE AS CODE
, a.KN AS NAME
, b.OPEN_P 
, b.HIGH_P
, b.LOW_P
, b.CLOSE_P 
from SECTOR..sb b,
SECTOR..sa a
where 1=1
and b.scode = 'A101'
and b.SCODE = a.SCODE
and b.td > '{from_date}'
