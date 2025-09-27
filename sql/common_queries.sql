

-- Getting orders data between two dates ('2025-09-01' AND '2025-09-27') for a user ('f17d6407-9d6b-45f5-854c-30a43b4b9615')

SELECT orders.order_id, orders.carrier, order_details.reference_number, order_details.customer_city, 
order_details.customer_state, order_details.warehouse_name, order_details.invoice_value, order_details.price_of_shipment, 
tracking_orders.carrier_name, tracking_orders.current_order_status, tracking_orders.current_order_status_code, 
tracking_orders.shipping_zone, tracking_orders.tracking_link, tracking_orders.awbno, tracking_orders.ordered_time, 
tracking_orders.expected_delivery_date, tracking_orders.pickup_date, tracking_orders.last_updated_time, 
TO_CHAR(tracking_orders.ordered_date, 'YYYY-MM-DD') as ordered_date, 
TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date 
FROM orders INNER JOIN order_details ON orders.order_id = order_details.order_id 
INNER JOIN tracking_orders ON order_details.order_id = tracking_orders.order_id 
WHERE orders.user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615' 
AND tracking_orders.ordered_date BETWEEN '2025-09-01' AND '2025-09-27' 
ORDER BY tracking_orders.ordered_date DESC



-- Fetching All orders ('CHANNEL_ORDER' means the order is not manifested yet)

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC


-- Fetching all delivered orders
WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.current_order_status = 'DELIVERED'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC


-- Fetching RTO orders:

 WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.current_order_status = ANY ('RTO', 'RTO IN TRANSIT', 'RTO DELIVERED')
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC


-- Orders from past 7 days:

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-09-20' AND '2025-09-27'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC
			

-- Undelivered orders from past 30 days

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-08-28' AND '2025-09-27'
			
				AND tracking_orders.current_order_status = 'UNDELIVERED'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC
	

-- Out for Delivery orders from past 7 days

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-09-20' AND '2025-09-27'
			
				AND tracking_orders.current_order_status = 'OUT FOR DELIVERY'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC
			

-- Ready for Pickup orders from today:

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date = '2025-09-27'
			
				AND tracking_orders.current_order_status = 'ORDERED'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC
			

-- COD orders from past 30 days:

 WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-08-28' AND '2025-09-27'
			
				AND order_details.order_type = 'COD'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC


-- In Transit Prepaid orders from past 7 days:

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-09-20' AND '2025-09-27'
			
				AND tracking_orders.current_order_status = ANY ('PICKED UP', 'IN TRANSIT')
			
				AND order_details.order_type = ANY ('PREPAID')
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC
			

-- RTO orders from Delhivery carrier for the past 30 days:

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-08-28' AND '2025-09-27'
			
				AND tracking_orders.current_order_status = ANY ('RTO', 'RTO IN TRANSIT', 'RTO DELIVERED')
			
				AND tracking_orders.carrier_name = 'Delhivery'
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC


-- In Transit and Out for Delivery orders from Delhivery and Blue Dart carriers from the past 7 days:

WITH user_orders AS (
			SELECT order_id, carrier 
			FROM orders 
			WHERE user_id = 'f17d6407-9d6b-45f5-854c-30a43b4b9615'
		)

		SELECT
			user_orders.order_id,
			user_orders.carrier,
			order_details.original_order_id,
			order_details.customer_city,
			tracking_orders.carrier_name,
			order_details.warehouse_name,
			order_details.customer_address,
			order_details.price_of_shipment,
			order_details.customer_email,
			order_details.order_type,
			order_details.order_mode,
			order_details.invoice_value,
			order_details.customer_name,
			order_details.customer_state,
			order_details.customer_phone,
			order_details.customer_pincode,
			order_details.order_items,
			order_details.dimensions,
			order_details.marketplace,
			order_details.gst_details,
			order_details.tax_percentage,
			order_details.hsn_code,
			order_details.invoice_number,
			order_details.ewaybill_serial_number, 
			order_details.order_note,
			order_details.is_hyperlocal_eligible,
			order_details.tags,
			tracking_orders.current_order_status,
			tracking_orders.current_order_status_code,
			tracking_orders.tracking_link,
			tracking_orders.awbno,
			tracking_orders.ordered_time,
			tracking_orders.pickup_date, 
			TO_CHAR(
				CASE
					WHEN tracking_orders.expected_delivery_date::text NOT IN ('DRAFTORDER', 'HYPERLOCAL_CONVERTED')
					THEN tracking_orders.expected_delivery_date::date
					ELSE NULL
				END,
				'DD Mon YYYY'
			) as expected_delivery_date,
			tracking_orders.last_updated_time,
			tracking_orders.sla_breach,
			TO_CHAR(tracking_orders.ordered_date, 'DD Mon YYYY') as ordered_date,
			TO_CHAR(tracking_orders.last_updated_date, 'YYYY-MM-DD') as last_updated_date,
			TO_CHAR(tracking_orders.pickup_date::date, 'DD Mon YYYY') as pickup_date
		FROM tracking_orders
		INNER JOIN user_orders ON tracking_orders.order_id = user_orders.order_id
		INNER JOIN order_details ON user_orders.order_id = order_details.order_id 
		WHERE 
			tracking_orders.current_order_status != 'CHANNEL_ORDER'
	
				AND tracking_orders.ordered_date BETWEEN '2025-09-20' AND '2025-09-27'
			
				AND tracking_orders.current_order_status = ANY ('PICKED UP', 'IN TRANSIT', 'OUT FOR DELIVERY')
			
				AND tracking_orders.carrier_name = ANY ('Delhivery', 'Blue Dart')
			
			ORDER BY
				tracking_orders.ordered_date DESC,
				tracking_orders.ordered_time DESC,
				order_details.original_order_id DESC


-- Carrier wise order count for user

SELECT 
    orders.carrier,
    COUNT(tracking_orders.awbno) AS awbno_count
FROM tracking_orders 
JOIN orders ON orders.order_id = tracking_orders.order_id 
WHERE orders.user_id = 'fba993c5-c349-4e17-a29b-41e9bd14ec04'
  AND tracking_orders.current_order_status NOT IN ('CANCELLED', 'CANCELLED NEW ORDERS', 'CHANNEL_ORDER')
  AND tracking_orders.awbno != 'DRAFTORDER'
  AND tracking_orders.ordered_date BETWEEN '2025-09-20' AND '2025-09-27'
GROUP BY orders.carrier
ORDER BY orders.carrier;


-- Pincode wise delivery rate

SELECT 
    od.customer_pincode,
    o.carrier,
    COUNT(*) FILTER (WHERE LOWER(t.current_order_status) = 'delivered') AS delivered_count,
    COUNT(*) FILTER (WHERE LOWER(t.current_order_status) IN ('rto delivered','rto','rto in transit')) AS returned_count,
	COUNT(*) FILTER (WHERE LOWER(t.current_order_status) IN ('delivered','rto delivered','rto','rto in transit')) AS total_count,
    ROUND(
        COUNT(*) FILTER (WHERE LOWER(t.current_order_status) = 'delivered')::numeric /
        NULLIF(
            COUNT(*) FILTER (
                WHERE LOWER(t.current_order_status) IN ('delivered','rto delivered','rto','rto in transit')
            ), 0
        ), 4
    ) AS delivery_rate
FROM tracking_orders t
JOIN order_details od 
    ON t.order_id = od.order_id
JOIN orders o
    ON t.order_id = o.order_id
WHERE LOWER(t.current_order_status) IN ('delivered','rto delivered','rto','rto in transit')
  AND od.customer_pincode ~ '^[0-9]{6}$'
   o.user_id = 'fba993c5-c349-4e17-a29b-41e9bd14ec04'
GROUP BY od.customer_pincode, o.carrier
ORDER BY od.customer_pincode, delivery_rate DESC;

