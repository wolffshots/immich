/* tslint:disable */
/* eslint-disable */
/**
 * Immich
 * Immich API
 *
 * The version of the OpenAPI document: 1.94.1
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


import * as runtime from '../runtime';
import type {
  CreateLibraryDto,
  LibraryResponseDto,
  LibraryStatsResponseDto,
  ScanLibraryDto,
  UpdateLibraryDto,
} from '../models/index';
import {
    CreateLibraryDtoFromJSON,
    CreateLibraryDtoToJSON,
    LibraryResponseDtoFromJSON,
    LibraryResponseDtoToJSON,
    LibraryStatsResponseDtoFromJSON,
    LibraryStatsResponseDtoToJSON,
    ScanLibraryDtoFromJSON,
    ScanLibraryDtoToJSON,
    UpdateLibraryDtoFromJSON,
    UpdateLibraryDtoToJSON,
} from '../models/index';

export interface CreateLibraryRequest {
    createLibraryDto: CreateLibraryDto;
}

export interface DeleteLibraryRequest {
    id: string;
}

export interface GetLibraryInfoRequest {
    id: string;
}

export interface GetLibraryStatisticsRequest {
    id: string;
}

export interface RemoveOfflineFilesRequest {
    id: string;
}

export interface ScanLibraryRequest {
    id: string;
    scanLibraryDto: ScanLibraryDto;
}

export interface UpdateLibraryRequest {
    id: string;
    updateLibraryDto: UpdateLibraryDto;
}

/**
 * 
 */
export class LibraryApi extends runtime.BaseAPI {

    /**
     */
    async createLibraryRaw(requestParameters: CreateLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<LibraryResponseDto>> {
        if (requestParameters.createLibraryDto === null || requestParameters.createLibraryDto === undefined) {
            throw new runtime.RequiredError('createLibraryDto','Required parameter requestParameters.createLibraryDto was null or undefined when calling createLibrary.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library`,
            method: 'POST',
            headers: headerParameters,
            query: queryParameters,
            body: CreateLibraryDtoToJSON(requestParameters.createLibraryDto),
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => LibraryResponseDtoFromJSON(jsonValue));
    }

    /**
     */
    async createLibrary(requestParameters: CreateLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<LibraryResponseDto> {
        const response = await this.createLibraryRaw(requestParameters, initOverrides);
        return await response.value();
    }

    /**
     */
    async deleteLibraryRaw(requestParameters: DeleteLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<void>> {
        if (requestParameters.id === null || requestParameters.id === undefined) {
            throw new runtime.RequiredError('id','Required parameter requestParameters.id was null or undefined when calling deleteLibrary.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library/{id}`.replace(`{${"id"}}`, encodeURIComponent(String(requestParameters.id))),
            method: 'DELETE',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.VoidApiResponse(response);
    }

    /**
     */
    async deleteLibrary(requestParameters: DeleteLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<void> {
        await this.deleteLibraryRaw(requestParameters, initOverrides);
    }

    /**
     */
    async getLibrariesRaw(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<Array<LibraryResponseDto>>> {
        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library`,
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => jsonValue.map(LibraryResponseDtoFromJSON));
    }

    /**
     */
    async getLibraries(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<Array<LibraryResponseDto>> {
        const response = await this.getLibrariesRaw(initOverrides);
        return await response.value();
    }

    /**
     */
    async getLibraryInfoRaw(requestParameters: GetLibraryInfoRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<LibraryResponseDto>> {
        if (requestParameters.id === null || requestParameters.id === undefined) {
            throw new runtime.RequiredError('id','Required parameter requestParameters.id was null or undefined when calling getLibraryInfo.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library/{id}`.replace(`{${"id"}}`, encodeURIComponent(String(requestParameters.id))),
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => LibraryResponseDtoFromJSON(jsonValue));
    }

    /**
     */
    async getLibraryInfo(requestParameters: GetLibraryInfoRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<LibraryResponseDto> {
        const response = await this.getLibraryInfoRaw(requestParameters, initOverrides);
        return await response.value();
    }

    /**
     */
    async getLibraryStatisticsRaw(requestParameters: GetLibraryStatisticsRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<LibraryStatsResponseDto>> {
        if (requestParameters.id === null || requestParameters.id === undefined) {
            throw new runtime.RequiredError('id','Required parameter requestParameters.id was null or undefined when calling getLibraryStatistics.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library/{id}/statistics`.replace(`{${"id"}}`, encodeURIComponent(String(requestParameters.id))),
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => LibraryStatsResponseDtoFromJSON(jsonValue));
    }

    /**
     */
    async getLibraryStatistics(requestParameters: GetLibraryStatisticsRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<LibraryStatsResponseDto> {
        const response = await this.getLibraryStatisticsRaw(requestParameters, initOverrides);
        return await response.value();
    }

    /**
     */
    async removeOfflineFilesRaw(requestParameters: RemoveOfflineFilesRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<void>> {
        if (requestParameters.id === null || requestParameters.id === undefined) {
            throw new runtime.RequiredError('id','Required parameter requestParameters.id was null or undefined when calling removeOfflineFiles.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library/{id}/removeOffline`.replace(`{${"id"}}`, encodeURIComponent(String(requestParameters.id))),
            method: 'POST',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.VoidApiResponse(response);
    }

    /**
     */
    async removeOfflineFiles(requestParameters: RemoveOfflineFilesRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<void> {
        await this.removeOfflineFilesRaw(requestParameters, initOverrides);
    }

    /**
     */
    async scanLibraryRaw(requestParameters: ScanLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<void>> {
        if (requestParameters.id === null || requestParameters.id === undefined) {
            throw new runtime.RequiredError('id','Required parameter requestParameters.id was null or undefined when calling scanLibrary.');
        }

        if (requestParameters.scanLibraryDto === null || requestParameters.scanLibraryDto === undefined) {
            throw new runtime.RequiredError('scanLibraryDto','Required parameter requestParameters.scanLibraryDto was null or undefined when calling scanLibrary.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library/{id}/scan`.replace(`{${"id"}}`, encodeURIComponent(String(requestParameters.id))),
            method: 'POST',
            headers: headerParameters,
            query: queryParameters,
            body: ScanLibraryDtoToJSON(requestParameters.scanLibraryDto),
        }, initOverrides);

        return new runtime.VoidApiResponse(response);
    }

    /**
     */
    async scanLibrary(requestParameters: ScanLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<void> {
        await this.scanLibraryRaw(requestParameters, initOverrides);
    }

    /**
     */
    async updateLibraryRaw(requestParameters: UpdateLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<LibraryResponseDto>> {
        if (requestParameters.id === null || requestParameters.id === undefined) {
            throw new runtime.RequiredError('id','Required parameter requestParameters.id was null or undefined when calling updateLibrary.');
        }

        if (requestParameters.updateLibraryDto === null || requestParameters.updateLibraryDto === undefined) {
            throw new runtime.RequiredError('updateLibraryDto','Required parameter requestParameters.updateLibraryDto was null or undefined when calling updateLibrary.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/library/{id}`.replace(`{${"id"}}`, encodeURIComponent(String(requestParameters.id))),
            method: 'PUT',
            headers: headerParameters,
            query: queryParameters,
            body: UpdateLibraryDtoToJSON(requestParameters.updateLibraryDto),
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => LibraryResponseDtoFromJSON(jsonValue));
    }

    /**
     */
    async updateLibrary(requestParameters: UpdateLibraryRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<LibraryResponseDto> {
        const response = await this.updateLibraryRaw(requestParameters, initOverrides);
        return await response.value();
    }

}
